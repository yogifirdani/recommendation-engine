import json
import hashlib
from database import get_active_packages, save_package_vector
from preprocessor import build_combined_features
from vectorizer import fit_and_save_vectorizer

def train():
    df_packages = get_active_packages()
    corpus = []
    package_ids = []
    for idx, row in df_packages.iterrows():
        combined_feat = build_combined_features(row)
        corpus.append(combined_feat)
        package_ids.append(int(row['id']))
        
    vectorizer, tfidf_matrix = fit_and_save_vectorizer(corpus)
    
    vocab_json_str = json.dumps(vectorizer.vocabulary_, sort_keys=True)
    vocab_hash = hashlib.sha256(vocab_json_str.encode('utf-8')).hexdigest()
    
    for i, pkg_id in enumerate(package_ids):
        vector_dense = tfidf_matrix[i].toarray()[0].tolist()
        combined_text = corpus[i]
        save_package_vector(pkg_id, combined_text, vector_dense, vocab_hash)
        
    print("Training Berhasil!")

if __name__ == "__main__":
    train()
