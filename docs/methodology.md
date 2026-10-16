# Methodology

This project uses RDKit to parse SMILES strings, calculate canonical and canonical isomeric SMILES, and generate molecular fingerprints/descriptors.

## Fingerprints and similarity

- Default fingerprint: Morgan radius 2, 2048 bits.
- Optional: radius 3 and bit sizes 1024/2048/4096.
- Similarity metric: Tanimoto coefficient on binary fingerprints.
- Similarity thresholds are user-defined analytical choices.

## Clustering

- Agglomerative clustering uses precomputed distances `D = 1 - T`.
- Butina clustering uses RDKit's distance-threshold approach.
- Clusters reflect structural grouping only and are not experimentally validated families.

## PCA and visualization

- Descriptor features are median-imputed and standardized before PCA.
- PCA is a low-dimensional projection and does not preserve all structure relationships.
