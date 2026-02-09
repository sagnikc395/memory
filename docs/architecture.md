# high level architecture of our memories

1. Extract relevant facts from user's messages
    - DSPy -> pass in conversations and ask LLMs for things that it finds interesting.
2. Uploading memories into the database
    - Add the memories as vector embeddings and factoids that the agent must remember.
    - Host QDrant locally to get data
3. Fetching relevant memories depending on query
4. Keep memories upto date
    - using the mem0 paper as a reference , decide to do operations (ADD, UPDATE,DELETE,NOOP)
