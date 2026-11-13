"""SRTP Handler for encrypted RTP audio.

Implements SRTP (Secure RTP) decryption for Siedle audio streams.
"""
import base64
import hashlib
import hmac
import logging
import os
import struct
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

_LOGGER = logging.getLogger(__name__)


# Crypto suites this module implements (RFC 4568 names).
SUPPORTED_SUITES = ("AES_CM_128_HMAC_SHA1_80",)

class SRTPCrypto:
    """Handle SRTP encryption/decryption."""
    
    def __init__(self, master_key: bytes, master_salt: bytes):
        """Initialize SRTP crypto.
        
        Args:
            master_key: 16-byte AES master key
            master_salt: 14-byte master salt
        """
        self.master_key = master_key
        self.master_salt = master_salt
        self._session_keys = {}
        # Rollover counter per SSRC, keeps the CTR keystream unique past the
        # 16-bit sequence wrap.
        self._roc_state: dict = {}
        # Integrity checking is enforced from the first tag that verifies.
        self._auth_verified_once = False
        self._auth_warned = False

        _LOGGER.debug(f"SRTP initialized: key={len(master_key)} bytes, salt={len(master_salt)} bytes")
    
    @classmethod
    def from_base64(cls, crypto_suite: str) -> Optional["SRTPCrypto"]:
        """Create SRTP crypto from SDP crypto attribute.
        
        Example: "AES_CM_128_HMAC_SHA1_80 inline:hVBnwvvAf8nCFBAI0vJRa7sTd38YeTkS22otYkiP"
        
        Args:
            crypto_suite: The crypto suite string from SDP
            
        Returns:
            SRTPCrypto instance or None if parsing failed
        """
        import base64
        
        try:
            # Extract key material from "inline:..." format
            if "inline:" not in crypto_suite:
                _LOGGER.error(f"Invalid crypto suite format: {crypto_suite}")
                return None
            
            # inline:<key||salt>[|lifetime][|MKI:length] [session params]
            key_material_b64 = crypto_suite.split("inline:", 1)[1].strip()
            key_material_b64 = key_material_b64.split()[0].split("|")[0]
            key_material = base64.b64decode(key_material_b64, validate=True)
            
            # AES_CM_128_HMAC_SHA1_80 uses:
            # - 16 bytes master key
            # - 14 bytes master salt
            # Total: 30 bytes
            if len(key_material) < 30:
                _LOGGER.error(f"Key material too short: {len(key_material)} bytes (expected 30)")
                return None
            
            master_key = key_material[:16]
            master_salt = key_material[16:30]
            
            _LOGGER.info(f"Parsed SRTP crypto: key={len(master_key)}B, salt={len(master_salt)}B")
            
            return cls(master_key, master_salt)
            
        except Exception as e:
            _LOGGER.error(f"Failed to parse crypto suite: {e}")
            return None
    
    @staticmethod
    def select_offer_crypto(sdp: str) -> Optional[Tuple[str, str]]:
        """Pick the a=crypto line of the audio stream we can handle.

        Returns (tag, attribute value) or None. Only the audio m-section is
        looked at (an offer may carry video with its own keys), and only
        AES_CM_128_HMAC_SHA1_80, the one suite this module implements. The
        old code took the first a=crypto line of the whole body, whatever
        its suite or media line.
        """
        in_audio = False
        for line in sdp.replace("\r\n", "\n").split("\n"):
            line = line.strip()
            if line.startswith("m="):
                in_audio = line.startswith("m=audio")
                continue
            if not in_audio or not line.startswith("a=crypto:"):
                continue
            value = line[len("a=crypto:"):]
            parts = value.split()
            if len(parts) >= 3 and parts[1] in SUPPORTED_SUITES:
                return parts[0], value
        return None

    @staticmethod
    def generate_sdp_crypto_line(tag: str = "1") -> Tuple[str, "SRTPCrypto"]:
        """Generate a new SRTP crypto line for SDP answer.
        
        Creates random 16-byte key + 14-byte salt (30 bytes total),
        base64 encodes them, and formats as SDP a=crypto: attribute value.
        Also returns the SRTPCrypto instance for encrypting outbound packets.
        
        Returns:
            Tuple of (sdp_line like "1 AES_CM_128_HMAC_SHA1_80 inline:<base64key>",
                       SRTPCrypto instance for encryption)
        """
        key_material = os.urandom(30)  # 16 byte key + 14 byte salt
        key_b64 = base64.b64encode(key_material).decode('ascii')
        # RFC 4568 6.1: the answer echoes the tag of the offer line it accepts.
        sdp_line = f"{tag} AES_CM_128_HMAC_SHA1_80 inline:{key_b64}"
        
        master_key = key_material[:16]
        master_salt = key_material[16:30]
        crypto = SRTPCrypto(master_key, master_salt)
        
        return sdp_line, crypto
    
    def _derive_session_key(self, label: int, index: int = 0) -> bytes:
        """Derive session key from master key using PRF.
        
        Args:
            label: Key derivation label (0x00=enc, 0x01=auth, 0x02=salt)
            index: Packet index (usually 0 for initial derivation)
            
        Returns:
            Derived session key
        """
        # Key derivation: AES-CM mode with master_key
        # Input: master_salt XOR (label || index || 0x00...)
        
        # Build IV for key derivation
        iv = bytearray(16)
        iv[0:14] = self.master_salt
        
        # XOR with label and index
        iv[7] ^= label
        
        # Encrypt zeros with AES-CM
        cipher = Cipher(
            algorithms.AES(self.master_key),
            modes.CTR(bytes(iv)),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Derive 16 bytes for encryption key, or 14 bytes for salt
        # RFC 3711: n_e = 128 bit, n_a = 160 bit (HMAC-SHA1), n_s = 112 bit
        key_len = {0x00: 16, 0x01: 20, 0x02: 14}[label]
        session_key = encryptor.update(b'\x00' * key_len)
        
        return session_key
    
    def get_session_keys(self, ssrc: int) -> Tuple[bytes, bytes, bytes]:
        """Get or derive session keys for a given SSRC.
        
        Args:
            ssrc: RTP SSRC identifier
            
        Returns:
            (session_key, session_auth_key, session_salt)
        """
        if ssrc in self._session_keys:
            return self._session_keys[ssrc]
        
        # Derive keys for this SSRC
        session_key = self._derive_session_key(0x00)  # Encryption key
        session_auth_key = self._derive_session_key(0x01)  # Auth key
        session_salt = self._derive_session_key(0x02)  # Salt key
        
        keys = (session_key, session_auth_key, session_salt)
        self._session_keys[ssrc] = keys
        
        _LOGGER.debug(f"Derived session keys for SSRC {ssrc:08x}")
        
        return keys
    
    def _build_iv(self, session_salt: bytes, header: bytes, ssrc: int,
                  sequence: int) -> bytes:
        """Build the AES-CTR IV, including a rollover counter.

        The 16-bit sequence number alone repeats after ~21 minutes of speech,
        and with it the CTR keystream — the classic two-time-pad problem. The
        rollover counter is mixed in so each packet keeps a unique counter
        block. It is XORed in, so for the first rollover period (ROC == 0) the
        IV is bit-for-bit what this implementation produced before, which
        keeps existing peers interoperable.
        """
        state = self._roc_state.setdefault(ssrc, {"roc": 0, "last_seq": sequence})
        if sequence < state["last_seq"] and state["last_seq"] - sequence > 32768:
            state["roc"] = (state["roc"] + 1) & 0xFFFFFFFF
        state["last_seq"] = sequence

        iv = bytearray(16)
        iv[0:14] = session_salt          # k_s * 2^16 (bytes 14..15 stay 0 = block counter)
        for i in range(4):               # XOR SSRC * 2^64 -> bytes 4..7
            iv[4 + i] ^= header[8 + i]
        index = (state["roc"] << 16) | sequence   # 48-bit packet index
        index_bytes = index.to_bytes(6, "big")
        for i in range(6):               # XOR i * 2^16 -> bytes 8..13
            iv[8 + i] ^= index_bytes[i]

        return bytes(iv)

    def decrypt_rtp(self, packet: bytes) -> Optional[bytes]:
        """Decrypt an SRTP packet to RTP.
        
        Args:
            packet: SRTP packet bytes (header + encrypted payload + auth tag)
            
        Returns:
            Decrypted RTP packet bytes, or None if decryption failed
        """
        try:
            # SRTP packet format:
            # [RTP Header (12+ bytes)] [Encrypted Payload] [Auth Tag (10 bytes)]
            
            if len(packet) < 12 + 10:  # Min header + auth tag
                _LOGGER.debug(f"Packet too short: {len(packet)} bytes")
                return None
            
            # Parse RTP header
            header = packet[:12]
            first_byte = header[0]
            cc = first_byte & 0x0F  # CSRC count
            
            # Calculate header length (skip CSRC if present)
            header_len = 12 + (cc * 4)
            
            if len(packet) < header_len + 10:
                _LOGGER.debug(f"Packet too short for header: {len(packet)} bytes")
                return None
            
            # Extract SSRC for key derivation
            ssrc = struct.unpack("!I", header[8:12])[0]
            
            # Get session keys
            session_key, session_auth_key, session_salt = self.get_session_keys(ssrc)
            
            # Extract auth tag (last 10 bytes for HMAC-SHA1-80)
            auth_tag = packet[-10:]
            srtp_packet = packet[:-10]  # Everything except auth tag

            # Verify the HMAC. This used to be skipped entirely, which left
            # SRTP without any integrity protection — anyone able to reach the
            # RTP port could inject audio without knowing the key.
            sequence = struct.unpack("!H", header[2:4])[0]
            iv = self._build_iv(session_salt, header, ssrc, sequence)
            roc = self._roc_state[ssrc]["roc"]
            expected_tag = hmac.new(
                session_auth_key, srtp_packet + struct.pack("!I", roc), hashlib.sha1
            ).digest()[:10]
            if hmac.compare_digest(expected_tag, auth_tag):
                self._auth_verified_once = True
            elif self._auth_verified_once:
                # We know this peer's tags normally verify, so this packet is
                # forged or corrupted.
                _LOGGER.warning("SRTP: dropping packet with invalid auth tag")
                return None
            elif not self._auth_warned:
                # Never saw a valid tag yet: the peer may compute the tag over
                # a different input. Don't break working audio over it, but say
                # so loudly once — integrity is not enforced in this state.
                self._auth_warned = True
                _LOGGER.warning(
                    "SRTP: auth tag does not verify; continuing without integrity "
                    "checking until a valid tag is seen"
                )

            # Extract encrypted payload
            encrypted_payload = srtp_packet[header_len:]

            # Decrypt using AES-CTR
            cipher = Cipher(
                algorithms.AES(session_key),
                modes.CTR(bytes(iv)),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decrypted_payload = decryptor.update(encrypted_payload)
            
            # Reconstruct RTP packet
            rtp_packet = header + decrypted_payload
            
            return rtp_packet
            
        except Exception as e:
            _LOGGER.error(f"SRTP decryption error: {e}")
            return None

    def encrypt_rtp(self, packet: bytes) -> Optional[bytes]:
        """Encrypt a plain RTP packet to SRTP.
        
        Args:
            packet: Plain RTP packet bytes (header + payload)
            
        Returns:
            SRTP packet bytes (header + encrypted payload + 10-byte auth tag), or None on error
        """
        try:
            if len(packet) < 12:
                return None
            
            # Parse RTP header
            header = packet[:12]
            first_byte = header[0]
            cc = first_byte & 0x0F
            header_len = 12 + (cc * 4)
            
            if len(packet) < header_len:
                return None
            
            # Extract SSRC for key derivation
            ssrc = struct.unpack("!I", header[8:12])[0]
            session_key, session_auth_key, session_salt = self.get_session_keys(ssrc)
            
            # Extract payload
            payload = packet[header_len:]
            
            # Build IV for encryption (same construction as decrypt)
            sequence = struct.unpack("!H", header[2:4])[0]
            iv = self._build_iv(session_salt, header, ssrc, sequence)

            # Encrypt using AES-CTR
            cipher = Cipher(
                algorithms.AES(session_key),
                modes.CTR(bytes(iv)),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            encrypted_payload = encryptor.update(payload)
            
            # Build authenticated portion: header + encrypted_payload
            auth_portion = header + encrypted_payload
            
            # Compute HMAC-SHA1-80 authentication tag
            roc = self._roc_state[ssrc]["roc"]
            auth_tag = hmac.new(
                session_auth_key, auth_portion + struct.pack("!I", roc), hashlib.sha1
            ).digest()[:10]
            
            return auth_portion + auth_tag
            
        except Exception as e:
            _LOGGER.error(f"SRTP encryption error: {e}")
            return None