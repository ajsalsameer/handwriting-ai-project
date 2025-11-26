from huggingface_hub import scan_cache_dir

def delete_model_by_name(model_name_fragment):
    # 1. Scan the cache (finds all models and their heavy files)
    print(f"Scanning cache for '{model_name_fragment}'...")
    cache_info = scan_cache_dir()
    
    # 2. Find the specific model
    repos_to_delete = []
    for repo in cache_info.repos:
        if model_name_fragment.lower() in repo.repo_id.lower():
            repos_to_delete.append(repo)

    if not repos_to_delete:
        print("❌ No matching model found in cache.")
        return

    # 3. Calculate space and prepare deletion
    for repo in repos_to_delete:
        print(f"FOUND: {repo.repo_id} ({repo.size_on_disk_str})")
        
        # This generates the instructions to delete the REAL files (blobs)
        delete_strategy = cache_info.delete_revisions(repo.revisions)
        
        print(f"🗑️  Deleting {len(repo.revisions)} revisions and associated blobs...")
        delete_strategy.execute() # <--- The magic command
        print("✅ DELETED. Space reclaimed.")

if __name__ == "__main__":
    # Change this string to delete different models
    delete_model_by_name("paligemma")