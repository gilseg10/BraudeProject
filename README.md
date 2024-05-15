# MaskGAE: Evaluation of edges readiness using masked graph autoencoders
![MaskGAE framwork](https://github.com/gilseg10/BraudeProject/assets/157500311/d815c454-4ff2-418f-a2c3-dfd89ec5488f)

## Summery
Welcome to the repository for our research project on Masked Graph Autoencoders (MaskGAE). This project explores the innovative application of masked autoencoding techniques to graph neural networks, aiming to address the challenges faced by traditional graph autoencoders by incorporating masked graph modeling (MGM).

Our project explores a new way to enhance how we learn from complex network data using a method called Masked Graph Autoencoding (MaskGAE). Unlike traditional approaches, MaskGAE improves learning by selectively hiding parts of the network (like connections between nodes) and then trying to figure out what was hidden using the remaining visible data.

Our main goal is to understand which parts of a network structure are effective and which parts are redundant (repeating themselves during the process). We do this by hiding certain parts of the network (sub-graphs), analyzing how the network behaves with these parts missing, and then attempting to reconstruct the hidden parts. By repeating this process, we can determine how well the network's structure is based on how accurate it can replicate the hidden elements consistently.

## Research Goals
Our research primarily focuses on:

**Improving Understanding of Graph Structures:** We investigate how MGM can enhance the capability of graph autoencoders to learn meaningful and robust graph representations.

**Evaluating Model Performance:** Through rigorous theoretical analysis and empirical testing, we evaluate the performance of MaskGAE across various graph-based tasks such as node classification and link prediction.

**Developing Robust Models:** By applying the MaskGAE framework, we aim to develop models that are not only effective in handling the complexities of graph data but also superior in performance when compared to existing methodologies.
