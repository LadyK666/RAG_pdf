From the contributor [hhh2210](https://github.com/hhh2210).
## Text Chunking
**Core Code**: `extract_hierarchical_entities` in `hirag/_op.py`

**Key Steps**:
- The function processes text chunks to extract entities and relationships
- It uses LLM prompts defined in `PROMPTS["hi_entity_extraction"]` for entity extraction
- Each chunk is processed to extract entities via `_process_single_content_entity`
- Embeddings are created for all extracted entities
- It also extracts relationships between entities via `_process_single_content_relation`

## Entity extraction
- Happens in `_process_single_content_entity` and `_process_single_content_relation` functions within `extract_hierarchical_entities`. These functions:
    - Use an LLM to extract entities with structured prompts
    - Extract entity attributes like name, type, description, and source
    - Store entities in a knowledge graph and vector database
- The extracted entity information is stored in the knowledge graph and processed by the `_handle_single_entity_extraction` function (line 165), which parses entity attributes from LLM output.

## GMM Clustering
**Core Code**: Functions in `hirag/_cluster_utils.py`

**Key Steps**:
- Uses `sklearn.mixture.GaussianMixture` for clustering
- Automatically determines optimal number of clusters with `get_optimal_clusters`
- Applies dimension reduction with UMAP before clustering
- Returns clusters as labels and probabilities

## Summarization of Entities
- For each cluster from GMM clustering, generates a prompt with all entities in the cluster
- Uses LLM to generate summary entities for the cluster
- Parses the LLM response to extract new higher-level entities and relationships
- Creates embeddings for these summary entities
- Adds these summaries to the next layer in the hierarchy

**Prompt Design**: The `summary_clusters` prompt instructs the LLM to:
- "Identify at least one attribute entity for the given entity description list"
- Generate entities matching types from the meta attribute list: `["organization", "person", "location", "event", "product", "technology", "industry", "mathematics", "social sciences"]`

