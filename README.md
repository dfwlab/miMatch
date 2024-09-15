# miMatch: A microbial metabolic background matching tool bolstering the causal link between microbiota and human disease in metagenomics

We developed a novel microbial metabolic background matching tool, namely miMatch (https://github.com/ddhmed/miMatch), an innovative metagenomic sample-matching tool that uses microbial metabolic background as a comprehensive reference for host-related variables and employs propensity score matching to build case-control pairs, even in the absence of host confounders. A user-friendly web server (https://www.biosino.org/iMAC/mimatch) has been established to promote the integration of multiple metagenomic cohorts, strengthening causal relationships in metagenomic research.

<img width="1023" alt="图片" src="https://user-images.githubusercontent.com/15136517/215239644-232227da-a44e-4441-abff-ce860674da11.png">


- miMatch.py is the core algorithm file of miMatch.
- config.ini is the configuration file for the algorithm.
- /SimulatedDataset provides several sets of simulated data.
- /curatedMetagenomicData provides several metagenomic studies from curatedMetagenomicData.
- /Analysis provides the original analysis codes.
- /Figures provides the code of figures in the manuscript.
- /Benchmark provides several sets of consistency evaluation benchmark.
- /Modeling provides results of Matched Sampling Random Forest (MSRF).
