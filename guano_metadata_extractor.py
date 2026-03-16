import struct
import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GuanoMetadataExtractor:
    """
    GUANO metadata extractor for WAV files
    Extracts embedded metadata from bat call recordings
    """
    
    @staticmethod
    def read_guano_metadata(filename: str) -> Optional[Dict[str, str]]:
        """
        Read GUANO metadata from WAV file
        
        Args:
            filename: Path to WAV file
            
        Returns:
            Dictionary of metadata key-value pairs, or None if not found
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"File not found: {filename}")
                return None

            with open(filename, 'rb') as f:
                # 1. Verify RIFF Header
                if f.read(4) != b'RIFF':
                    logger.error("Not a valid RIFF file")
                    return None
                
                f.read(4)  # Skip file size
                if f.read(4) != b'WAVE':
                    logger.error("Not a WAVE file")
                    return None
                
                # 2. Iterate chunks to find 'guan'
                while True:
                    try:
                        chunk_header = f.read(8)
                        if len(chunk_header) < 8:
                            break  # End of file
                        
                        chunk_id, chunk_size = struct.unpack('<4sI', chunk_header)
                        
                        if chunk_id == b'guan':
                            logger.info(f"Found 'guan' chunk (Size: {chunk_size} bytes)")
                            
                            # Read the metadata content
                            guano_data = f.read(chunk_size)
                            
                            # Decode UTF-8 text
                            try:
                                guano_text = guano_data.decode('utf-8', errors='replace')
                                
                                # Parse line by line into dictionary
                                metadata = {}
                                for line in guano_text.split('\n'):
                                    line = line.strip()
                                    if line and ':' in line:
                                        key, value = line.split(':', 1)
                                        metadata[key.strip()] = value.strip()
                                
                                logger.info(f"Extracted {len(metadata)} metadata fields")
                                return metadata
                                
                            except Exception as e:
                                logger.error(f"Error decoding GUANO text: {e}")
                                return None
                        else:
                            # Skip other chunks (fmt, data, etc.)
                            f.seek(chunk_size, 1)
                            
                            # Handle padding byte if chunk size is odd
                            if chunk_size % 2 == 1:
                                f.seek(1, 1)
                                
                    except struct.error:
                        break
                
                logger.warning("No 'guan' metadata chunk found in file")
                return None
                
        except Exception as e:
            logger.error(f"Error reading GUANO metadata: {e}")
            return None
    
    @staticmethod
    def extract_key_parameters(metadata: Optional[Dict[str, str]]) -> Dict[str, any]:
        """
        Extract key parameters from GUANO metadata
        
        Args:
            metadata: Dictionary of GUANO metadata
            
        Returns:
            Dictionary of formatted key parameters
        """
        if not metadata:
            return {
                'timestamp': None,
                'latitude': None,
                'longitude': None,
                'temperature': None,
                'humidity': None,
                'species': None,
                'length': None,
                'sample_rate': None,
                'filter_hp': None,
                'filter_lp': None,
                'make': None,
                'model': None,
                'firmware': None,
                'note': None
            }
        
        # Extract location coordinates properly
        loc_position = metadata.get('Loc Position') or metadata.get('GUANO|Loc Position')
        latitude = None
        longitude = None
        if loc_position:
            try:
                parts = loc_position.strip().split()
                if len(parts) >= 2:
                    latitude = parts[0]
                    longitude = parts[1]
            except Exception:
                pass
        
        # Fallback: check for standalone Latitude/Longitude keys (e.g. Dhvani devices)
        if not latitude:
            latitude = metadata.get('Latitude') or metadata.get('GUANO|Latitude')
        if not longitude:
            longitude = metadata.get('Longitude') or metadata.get('GUANO|Longitude')
        
        return {
            'timestamp': metadata.get('Timestamp') or metadata.get('GUANO|Timestamp'),
            'latitude': latitude,
            'longitude': longitude,
            'temperature': metadata.get('Temperature Ext') or metadata.get('GUANO|Temperature Ext'),
            'humidity': metadata.get('Humidity') or metadata.get('GUANO|Humidity'),
            'species': metadata.get('Species Manual ID') or metadata.get('GUANO|Species Manual ID'),
            'length': metadata.get('Length') or metadata.get('GUANO|Length'),
            'sample_rate': metadata.get('TE') or metadata.get('GUANO|TE'),
            'filter_hp': metadata.get('Filter HP') or metadata.get('GUANO|Filter HP'),
            'filter_lp': metadata.get('Filter LP') or metadata.get('GUANO|Filter LP'),
            'make': metadata.get('Make') or metadata.get('GUANO|Make'),
            'model': metadata.get('Model') or metadata.get('GUANO|Model'),
            'firmware': metadata.get('Firmware Version') or metadata.get('GUANO|Firmware Version'),
            'note': metadata.get('Note') or metadata.get('GUANO|Note'),
            'raw_metadata': metadata  # Include full metadata for reference
        }


def extract_metadata_from_file(audio_path: str) -> Dict[str, any]:
    """
    Convenience function to extract metadata from audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary of extracted metadata
    """
    extractor = GuanoMetadataExtractor()
    raw_metadata = extractor.read_guano_metadata(audio_path)
    return extractor.extract_key_parameters(raw_metadata)
