import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_policies(policy_dir: str):
    """
    Load policy documents from the specified directory.

    Args:
        policy_dir (str): Directory containing policy text files.
    """
    sections = []
    for file in Path(policy_dir).glob("*.json"):
        try:
            with open(file, "r") as f:
                policy = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON file: {file}")
            continue
        except Exception as e:
            logger.warning(f"Failed to read file {file}: {e}")
            continue

        policy_name = policy.get("policy", "Unknown Policy")
        policy_sections = policy.get("sections", [])
        if not policy_sections:
            logger.info(f"No sections found in policy: {policy_name}")
            continue

        for section in policy_sections:
            section_id = section.get("section", "Unknown Section")
            title = section.get("title", "No Title")
            text = section.get("text", "")
            if not text:
                logger.info(f"Empty text in section: {section_id} of policy: {policy_name}")
                continue
            sections.append({
                "policy": policy_name,
                "section": section_id,
                "title": title,
                "text": text
            })
            
    return sections


# Test usage
if __name__ == "__main__":
    policy_directory = "./data/raw_docs"
    loaded_sections = load_policies(policy_directory)
    for section in loaded_sections:
        print(section)