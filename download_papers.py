
import arxiv
import os
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def download_papers(paper_ids, download_path):
    """
    Downloads papers from arXiv based on a list of paper IDs.

    Args:
        paper_ids (list): A list of strings, where each string is an arXiv paper ID.
        download_path (str): The directory where papers will be saved.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)
        print(f"Created directory: {download_path}")

    print(f"Downloading {len(paper_ids)} papers to {download_path}...")

    client = arxiv.Client()
    
    search = arxiv.Search(id_list=paper_ids)
    
    for result in client.results(search):
        pdf_filename = f"{result.entry_id.split('/')[-1]}.pdf"
        print(f"Downloading '{result.title}' as {pdf_filename}...")
        result.download_pdf(dirpath=download_path, filename=pdf_filename)
    
    print("All papers downloaded successfully.")


if __name__ == "__main__":
    paper_collections = {
        "llm": ["1706.03762", "1810.04805", "2307.09288"], # Attention, BERT, Llama 2
        "cv": ["1512.03385", "1506.02640", "1409.1556"]   # ResNet, U-Net, VGG16
    }

    parser = argparse.ArgumentParser(description="Download papers from arXiv.")
    parser.add_argument("--collection", type=str, required=True, choices=paper_collections.keys(),
                        help="The paper collection to download.")
    
    args = parser.parse_args()

    if args.collection == "llm":
        download_path = "data/large_language_models"
        ids_to_download = paper_collections["llm"]
    elif args.collection == "cv":
        download_path = "data/computer_vision"
        ids_to_download = paper_collections["cv"]
    
    download_papers(ids_to_download, download_path)