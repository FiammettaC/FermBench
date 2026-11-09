# FermBench

### Project Abstract
Generative Artificial Intelligence (GenAI) chatbots, such as ChatGPT, Gemini, and Copilot, continue to amaze users worldwide on their quick developments and improvements. 
These tools posses vast general knowledge and thus could be used in different fields and applications. 
One of these is education, where these tools could be used to support students with their learning process and teachers in the development of material and instant student aid. 
However, now that many GenAI chatbots are available, how can we select the best one for domain-specific tasks? This study proposes a large-scale benchmark of five commercially available GenAI chatbots on the topic of fermentation. 
We introduce FermBench, a benchmark dataset tailored to fermentation processes. 
The performances of the chatbots are also compared against field experts. 
The experts are first asked to answer the questions contained in the dataset, then to rank the responses of the GenAI chatbots, and ultimately to provide a final, curated response to the given question. 
The curated responses are then used to enrich the knowledge of a previously implemented, open-source GenAI chatbot, FermentAI. 
This chatbot, which knowledge is extended with the collected curated answers, is deployed in BioVL, an e-learning platform aiming to teach students about fermentation processes. 
The objective of this study is to provide a thorough investigation of which commercially available GenAI chatbot performs better on the topic of fermentation, aiding students and teachers to pick the best one when needing help on the topic.

# Updates
Check our updated results [here](https://github.com/FiammettaC/FermBench-ESCAPE).

## Cite this work
If you would like to cite this work, please reference our paper: **SPyCE: A structured and tailored series of Python courses for (bio)chemical engineers**.
```bibtex
@article{caccavale2026fermbench,
  title={FermBench: a new benchmark for measuring the capabilities of LLMs on fermentation knowledge},
  author={Caccavale, Fiammetta and Aouichaoui, Adem RN and Kr{\"u}hne, Ulrich and Gernaey, Krist V and Gargalo, Carina L},
  journal={Computers and Education: Artificial Intelligence},
  pages={100577},
  year={2026},
  publisher={Elsevier}
}
```
