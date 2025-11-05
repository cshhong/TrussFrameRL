import json
import re
import os

def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_variable_mappings(config):
    """Create mappings between config keys and HTML variables"""
    
    # Direct mappings (config key -> HTML variable)
    direct_mappings = {}
    
    # Add all config variables as direct mappings first
    for key, value in config.items():
        if value and str(value).strip():
            direct_mappings[key] = str(value)
    
    # Handle special HTML variable formats
    html_mappings = {
        # ArXiv ID variations
        '<ARXIV PAPER ID>': config.get('ARXIV_PAPER_ID', ''),
        'ARXIV PAPER ID': config.get('ARXIV_PAPER_ID', ''),
        
        # Personal links (spaces in HTML)
        'FIRST AUTHOR PERSONAL LINK': config.get('FIRST_AUTHOR_PERSONAL_LINK', ''),
        'SECOND AUTHOR PERSONAL LINK': config.get('SECOND_AUTHOR_PERSONAL_LINK', ''),
        'THIRD AUTHOR PERSONAL LINK': config.get('THIRD_AUTHOR_PERSONAL_LINK', ''),
        
        # GitHub repo
        'YOUR REPO HERE': config.get('YOUR_REPO_HERE', ''),
        
        # Domain variations
        'YOUR_DOMAIN.com': config.get('YOUR_DOMAIN', '').replace('https://', '').replace('http://', ''),
        'YOUR_INSTITUTION_WEBSITE.com': config.get('YOUR_INSTITUTION_WEBSITE', ''),
        
        # Twitter handles (add @ if needed)
        '@YOUR_TWITTER_HANDLE': config.get('YOUR_TWITTER_HANDLE', '').lstrip('@'),
        '@AUTHOR_TWITTER_HANDLE': config.get('AUTHOR_TWITTER_HANDLE', '').lstrip('@'),
    }
    
    # Add HTML mappings to direct mappings
    for html_var, config_val in html_mappings.items():
        if config_val:
            direct_mappings[html_var] = config_val
    
    return direct_mappings

def handle_content_mapping(content, config):
    """Handle mapping between config variables and HTML content variables"""
    
    print("--- Content Section Mapping ---")
    
    # Content mappings (HTML text -> config value)
    content_mappings = {
        # Main title
        'Academic Project Page': config.get('PAPER_TITLE', ''),
        
        # Author names in content section
        'First Author': config.get('FIRST_AUTHOR_NAME', ''),
        'Second Author': config.get('SECOND_AUTHOR_NAME', ''), 
        'Third Author': config.get('THIRD_AUTHOR_NAME', ''),
        
        # Institution and conference
        'Institution Name': config.get('INSTITUTION_OR_LAB_NAME', ''),
        'Conference name and year': config.get('CONFERENCE_NAME', ''),
        
        # Abstract placeholder (long text)
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin ullamcorper tellus sed ante aliquam tempus. Etiam porttitor urna feugiat nibh elementum, et tempor dolor mattis. Donec accumsan enim augue, a vulputate nisi sodales sit amet. Proin bibendum ex eget mauris cursus euismod nec et nibh. Maecenas ac gravida ante, nec cursus dui. Vivamus purus nibh, placerat ac purus eget, sagittis vestibulum metus. Sed vestibulum bibendum lectus gravida commodo. Pellentesque auctor leo vitae sagittis suscipit.': config.get('FULL_ABSTRACT_TEXT_HERE', ''),
        
        # Video description
        'Aliquam vitae elit ullamcorper tellus egestas pellentesque. Ut lacus tellus, maximus vel lectus at, placerat pretium mi. Maecenas dignissim tincidunt vestibulum. Sed consequat hendrerit nisl ut maximus.': config.get('VIDEO_DESCRIPTION', ''),
        
        # BibTeX placeholders
        'YourPaperKey2024': config.get('BIBTEX_ID', ''),
        'Your Paper Title Here': config.get('PAPER_TITLE', ''),
        'First Author and Second Author and Third Author': config.get('BIBTEX_AUTHORS', ''),
        'Conference/Journal Name': config.get('CONFERENCE_OR_JOURNAL_NAME', ''),
        '2024': config.get('PUBLICATION_YEAR', ''),
        'https://your-domain.com/your-project-page': config.get('YOUR_PROJECT_URL', ''),
        
        # Image carousel descriptions
        'First image description.': config.get('IMAGE_1_DESCRIPTION', ''),
        'Second image description.': config.get('IMAGE_2_DESCRIPTION', ''),
        'Third image description.': config.get('IMAGE_3_DESCRIPTION', ''),
        'Fourth image description.': config.get('IMAGE_4_DESCRIPTION', ''),
    }
    
    # Apply content mappings (only replace if config value exists and is not empty)
    for html_text, config_value in content_mappings.items():
        if config_value and str(config_value).strip():
            content = content.replace(html_text, str(config_value))
            print(f"✓ Content: '{html_text[:40]}...' -> '{str(config_value)[:50]}{'...' if len(str(config_value)) > 50 else ''}'")
    
    return content

def handle_special_cases(content, config):
    """Handle special URL construction and formatting cases"""
    
    print("\n--- Special Cases ---")
    
    # Handle YouTube video ID
    if 'YOUTUBE_VIDEO_ID' in config and config['YOUTUBE_VIDEO_ID']:
        old_id = 'JkaxUblCGz0'
        new_id = config['YOUTUBE_VIDEO_ID']
        content = content.replace(old_id, new_id)
        print(f"✓ YouTube ID: {old_id} -> {new_id}")
    
    # Handle Twitter handles (add @ if not present)
    twitter_fields = ['YOUR_TWITTER_HANDLE', 'AUTHOR_TWITTER_HANDLE']
    for field in twitter_fields:
        if field in config and config[field]:
            handle = config[field]
            if not handle.startswith('@'):
                handle = '@' + handle
            # Replace both @FIELD and FIELD patterns
            content = content.replace(f'@{field}', handle)
            content = content.replace(field, handle.lstrip('@'))
            print(f"✓ Twitter: {field} -> {handle}")
    
    # Fix double protocols and URLs
    content = re.sub(r'https://https://', 'https://', content)
    content = re.sub(r'http://https://', 'https://', content)
    
    # Fix double github in URLs
    content = re.sub(r'github\.com/https://github\.com/', 'github.com/', content)
    
    return content

def generate_index_html(template_file='_index.html', output_file='index.html', config_file='config.json'):
    """Generate index.html from template using config values"""
    
    # Check if files exist
    if not os.path.exists(template_file):
        print(f"❌ Error: Template file '{template_file}' not found")
        return False
    
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file '{config_file}' not found")
        return False
    
    # Load configuration
    try:
        config = load_config(config_file)
        print(f"✓ Loaded config with {len(config)} variables")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return False
    
    # Read template
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✓ Loaded template file: {template_file}")
    except Exception as e:
        print(f"❌ Error reading template file: {e}")
        return False
    
    print(f"\n--- Variable Replacements ---")
    
    # Create variable mappings
    mappings = create_variable_mappings(config)
    
    # Replace all variables
    replacements_made = 0
    for html_var, config_val in mappings.items():
        if config_val and str(config_val).strip():
            old_content = content
            content = content.replace(html_var, str(config_val))
            if content != old_content:
                replacements_made += 1
                print(f"✓ {html_var} -> {str(config_val)[:60]}{'...' if len(str(config_val)) > 60 else ''}")
    
    # Handle content section mapping
    content = handle_content_mapping(content, config)
    
    # Handle special cases
    content = handle_special_cases(content, config)
    
    # Write the updated content
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Successfully generated {output_file}")
        print(f"📊 Made {replacements_made} variable replacements")
        return True
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate index.html from template using config.json')
    parser.add_argument('--template', default='_index.html', help='Template HTML file')
    parser.add_argument('--output', default='index.html', help='Output HTML file')
    parser.add_argument('--config', default='config.json', help='Config JSON file')
    
    args = parser.parse_args()
    
    print("🚀 Academic Project Page Generator")
    print("=" * 50)
    
    success = generate_index_html(args.template, args.output, args.config)
    
    if success:
        print(f"\n🌐 Open http://localhost:8080 to view your page")
        print(f"📁 Add your media files to the static/ folder")
    else:
        print(f"\n💡 Check the errors above and try again")