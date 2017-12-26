# hierarchy-rnn-sentence-match

This neural network could solve context-sensitive response selection task in multi-turn retrieval based chatbot.

For example, you have a current user query and the conversation history:

c1: How good is Justice league in 2017?
answer: ...
c2: Who stars Superman?
answer: ...
q: How about Batman?

Your chatbot is able to select correct response from the candidate pool.
r1: It is Ben Affleck
r2: Batman is Bruce Wayne.

This project implements a hierarchy rnn to do this task. For each sentence in conversation history, a RNN is used to encode the sentence to fixed length vector. Then another RNN is used to encode sentences' vector to one vector which capture the semantic meaning of the whole conversation history. Each response is also encoded by a rnn into one vector. Then the similarity between conversation history vector and response vector is caculated by a fully connected layer to output the final rank score.
