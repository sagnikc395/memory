import litellm

# to check how groq models are namespaced
print(litellm.models_by_provider.get("groq"))
