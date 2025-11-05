import json
import re

def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def update_html_template(template_file='index.html', output_file='index_updated.html', config_file='config.json'):
    """Update HTML template with values from config file"""
    
    # Load configuration
    config = load_config(config_file)
    
    # Read template
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the mapping of placeholders to config keys
    # This handles variations in placeholder naming
    placeholder_mappings = {
        'PAPER_TITLE': 'paper_title',
        'AUTHOR_NAMES': 'authors',
        'FIRST_AUTHOR_NAME': 'first_author_name',
        'SECOND_AUTHOR_NAME': 'second_author_name',
        'THIRD_AUTHOR_NAME': 'third_author_name',
        'INSTITUTION_OR_LAB_NAME': 'institution_or_lab_name',
        'CONFERENCE_NAME': 'conference_name',
        'VENUE_NAME': 'venue_name',
        'BRIEF_DESCRIPTION_OF_YOUR_RESEARCH_CONTRIBUTION_AND_FINDINGS': 'brief_description',
        'DETAILED_DESCRIPTION': 'detailed_description',
        'YOUR_DOMAIN.com': 'domain',
        'YOUR_GITHUB_USERNAME': 'github_username',
        'GITHUB_REPO_NAME': 'github_repo',
        'ARXIV PAPER ID': 'arxiv_id',
        'ARXIV_ID': 'arxiv_id',
        'VIDEO_ID': 'video_id',
        'YOUTUBE_VIDEO_ID': 'video_id',
        'CONTACT_EMAIL': 'contact_email',
        'PUBLICATION_YEAR': 'bibtex_year',
        'CONFERENCE_YEAR': 'conference_year'
    }
    
    # Replace placeholders
    for placeholder, config_key in placeholder_mappings.items():
        if config_key in config:
            content = content.replace(placeholder, str(config[config_key]))
    
    # Handle special cases that might need URL construction
    if 'github_username' in config and 'github_repo' in config:
        github_url = f"https://github.com/{config['github_username']}/{config['github_repo']}"
        content = content.replace('GITHUB_URL', github_url)
    
    if 'arxiv_id' in config:
        arxiv_url = f"https://arxiv.org/abs/{config['arxiv_id']}"
        content = content.replace('ARXIV_URL', arxiv_url)
    
    if 'domain' in config and 'github_repo' in config:
        base_url = f"https://{config['domain']}/{config['github_repo']}"
        content = content.replace('BASE_URL', base_url)
        content = content.replace('YOUR_DOMAIN.com', config['domain'])
    
    # Write updated content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"HTML template updated successfully! Output saved to {output_file}")
    print(f"Replaced {len(placeholder_mappings)} different placeholder types")

if __name__ == "__main__":
    update_html_template()