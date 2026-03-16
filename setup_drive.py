#!/usr/bin/env python3
"""
Google Drive Setup Script for BCIT Admin Panel

This script helps you set up Google Drive API credentials.
Run this script to authenticate with Google Drive for the first time.
"""

import os
import json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def check_client_secrets():
    """Check if client_secrets.json exists and is valid"""
    if not os.path.exists("client_secrets.json"):
        print("❌ client_secrets.json not found!")
        print("\nTo create this file:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable Google Drive API")
        print("4. Go to 'Credentials' → 'Create Credentials' → 'OAuth 2.0 Client ID'")
        print("5. Choose 'Desktop application' as application type")
        print("6. Download the JSON file and rename it to 'client_secrets.json'")
        print("7. Place it in the backend folder")
        return False
    
    try:
        with open("client_secrets.json", "r") as f:
            secrets = json.load(f)
            
        if "installed" in secrets or "web" in secrets:
            print("✅ client_secrets.json found and appears valid")
            return True
        else:
            print("❌ client_secrets.json format is invalid")
            return False
            
    except json.JSONDecodeError:
        print("❌ client_secrets.json is not valid JSON")
        return False

def setup_drive():
    """Set up Google Drive authentication"""
    try:
        print("🔧 Setting up Google Drive authentication...")
        
        gauth = GoogleAuth()
        
        # Try to load saved credentials
        if os.path.exists("credentials.json"):
            gauth.LoadCredentialsFile("credentials.json")
            
        if gauth.credentials is None:
            # Authenticate if credentials are not available
            print("🌐 Opening browser for authentication...")
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh credentials if expired
            print("🔄 Refreshing expired credentials...")
            gauth.Refresh()
        else:
            # Initialize the saved credentials
            gauth.Authorize()
            
        # Save the current credentials to file
        gauth.SaveCredentialsFile("credentials.json")
        
        # Test the connection
        drive = GoogleDrive(gauth)
        
        # Try to list some files to verify connection
        print("🔍 Testing connection...")
        file_list = drive.ListFile({'maxResults': 1}).GetList()
        
        print("✅ Google Drive setup successful!")
        print(f"✅ Found {len(file_list)} files in your Google Drive")
        print("✅ Credentials saved to credentials.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Google Drive: {e}")
        return False

def main():
    print("🚀 BCIT Admin Panel - Google Drive Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the backend directory")
        return
    
    # Check client secrets
    if not check_client_secrets():
        return
    
    # Set up Google Drive
    if setup_drive():
        print("\n🎉 Setup complete! You can now run:")
        print("   python app.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
