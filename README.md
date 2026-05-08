# neuro-AI

This project aims to implement a biologically inspired Recurrent Neural Network (RNN) model and compare its performance across neurogym tasks against a standard vanilla RNN. Specifically, the bio-RNN variant has its hidden state output modified so that it resembles the excitatory (**E**) vs inhibitory (**I**) effect of neurons found in real neural circuits (Dale's law). The control loop for the vanilla RNN variant (left) and the bio-RNN variant (right) are as follows:

<br>

<img width="800" height="360" alt="image" src="https://github.com/user-attachments/assets/688f2b7b-3542-418e-83b2-d4eeab67f98a" />

<br>

The experiments found no significant difference in performance between the two RNN variants. However, a clear neuronal firing pattern emerged from bio-RNN throughout the experiments: *signals fired by inhibitory neurons were significantly stronger in magnitude compared to their excitatory counterparts.* This effect may be explained by the ratio of **E**/**I** neurons implemented (4:1): since the inhibitory population is smaller, each neuron has to fire more strongly to maintain a balanced, stabilised neural state. This finding agrees with studies on real cortical circuits such as [Kajiwara et al.'s](https://pmc.ncbi.nlm.nih.gov/articles/PMC8031186/), where the stabilising ability of **I** neurons can be attributed to their higher firing rates, along with other topological factors.
