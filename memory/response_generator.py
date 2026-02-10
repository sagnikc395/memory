import dspy


class ResponseGenerator(dspy.Signature):
    history: dspy.History = dspy.InputField()
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    response: str = dspy.OutputField()


response_generator = dspy.Predict(ResponseGenerator)

