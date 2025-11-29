# src/index/sample_index.py

# Sample policy documents for RAG system testing
# Temporary synthetic data representing policy documents

sample_docs = sample_docs = [
    {
        "policy": "Policy: Domain Suspension Guidelines, Section 4.2",
        "text": "Domains may be suspended due to missing WHOIS info or violations of domain policies. Users must update WHOIS info to reactivate."
    },
    {
        "policy": "Policy: Password Reset Procedure, Step 1",
        "text": "Users can reset passwords via the login page by following the standard reset procedure."
    },
    {
        "policy": "Policy: Refund Policy, Section 1.5",
        "text": "Refunds are processed within 5 business days for canceled orders."
    }
]

def get_all_docs():
    """
    Returns all sample policy documents as a list of dictionaries.

    Each entry contains:
    - 'policy': The policy name and section.
    - 'text': The content of the policy.
    """

    return sample_docs