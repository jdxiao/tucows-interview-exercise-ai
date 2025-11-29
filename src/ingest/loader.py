import json
from pathlib import Path

def load_policies(policy_dir: str):
    """
    Load policy documents from the specified directory.

    Args:
        policy_dir (str): Directory containing policy text files.
    """
    sections = []
    for file in Path(policy_dir).glob("*.json"):
        with open(file, "r") as f:
            policy = json.load(f)
            for section in policy.get("sections", []):
                sections.append({
                    "policy": policy["policy"],
                    "section": section["section"],
                    "title": section["title"],
                    "text": section["text"]
                })
    return sections


# Test usage
if __name__ == "__main__":
    policy_directory = "./data/raw_docs"
    loaded_sections = load_policies(policy_directory)
    for section in loaded_sections:
        print(section)