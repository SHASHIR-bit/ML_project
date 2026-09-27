# Entity Resolution Methodology
Team Name: Vector Strike

## 1. Preprocessing
We utilized PyArrow string compute functions (C++ backend) directly on Pandas dataframes to normalize 40 million business names and addresses in under 2 minutes. Our normalization included:
- Lowercasing and whitespace trimming
- Punctuation removal (Regex `[^a-z0-9\s]`)
- Prefix extraction (first 5 characters) and First Word extraction
- Pincode extraction from address (`\b\d{5,6}\b`)

## 2. Blocking (Candidate Generation)
To reduce 22 trillion possible comparisons, we applied a multi-strategy O(N) inverted-index blocking approach partitioned by Country:
1. Exact Name Prefix (first 5 chars) match
2. Exact First Word match
3. Exact Pincode match
4. Rare Token match (tokens with frequency < 300)

We capped maximum candidates per entity to 100 to maintain strict 16GB memory limits.

## 3. Feature Engineering
For each candidate pair, we computed 27 distinct similarity features including:
- Jaccard similarity of name/address tokens
- Character n-gram overlaps (2, 3, 4 grams)
- Token length ratios and exact matches
- Levenshtein distance proxies

## 4. Modeling
We trained a LightGBM Binary Classifier on the 10 million candidate pairs using an NVIDIA RTX 4050 GPU. 
- Objective: `binary_logloss`
- Max Depth: 8, Num Leaves: 63
- To maximize the precision-heavy F_0.5 metric, we applied `scale_pos_weight` during training to combat class imbalance, and ran a grid search on the validation set to dynamically select the optimal probability threshold (0.65). 
- Singletons are gracefully handled if no candidate exceeds the threshold.
