<div align="center">
  <img src="data/logo.png" width="300">
</div>

# SQARL: Scalable Qubit Allocation via Reinforcement Learning

The scaling of quantum processors is currently limited by technical challenges such as decoherence and cross-talk.
As the number of qubits grows, the interference between them adds more noise to the computations.
Distributed quantum computing addresses this limitation by interconnecting smaller, easier-to-handle quantum processors (cores), but it introduces the challenge of minimizing slow, error-prone inter-core communication.
The task of distributing quantum circuits across cores while minimizing communication costs is known as the~\emph{Qubit Allocation} problem.
This work focuses on developing a deep learning approach to this problem, emphasizing flexibility to quantum hardware topology and improving state-of-the-art performance.

On the one hand, heuristic and non-learning algorithms, such as the Hungarian Qubit Allocation (HQA), currently represent the state of the art.
On the other hand, Reinforcement Learning (RL) approaches leverage learned allocation policies, but are currently unable to match the allocation costs of HQA and often lack flexibility, that is, they require retraining when hardware configurations change (number of cores, number of qubits, or core interconnection costs), and fall short of the solution quality achieved by non-learning methods.
To overcome these limitations, this work proposes a flexible, transformer-based architecture that can handle arbitrary numbers of qubits and cores without retraining.

The results show that the trained policy achieves up to a 33% reduction in allocation cost relative to the state-of-the-art (HQA) for the Cuccaro Adder and narrows the gap between RL and HQA for the most common circuits.
These findings show that learning-based approaches can effectively match the performance of
hand-crafted heuristics, a crucial step towards their application in real-world scenarios.

## How to setup the project

Install the requirements via
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
and follow the notebook `src/tutorial.ipynb` for guidelines on how to optimize circuits and train allocators.