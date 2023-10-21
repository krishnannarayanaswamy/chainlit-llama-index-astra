A demo that demonstrates Llama Index, Chainlit interface for a chatbot and Data Astra as Vector Store.

Create Astra account, create vector database, download the secure bundle and create a token
Create a folder named config
Store the Astra token in json format
Store the secure conenct bundle


pip3 install llama-index cassandra-driver pypdf python-dotenv chainlit

export OPENAI_API_KEY='with your key'

chainlit run chainlitBotAstra.py -w