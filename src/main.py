import torch
from langchain.callbacks.manager import CallbackManager
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used for LLM: %s" % device)
    
    #print("Loading Embeddings...")
    #embedding = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})
    #print("Loaded: %s" % embedding)

    print("Loading Handler")

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path= "C:\LLM\codellama-7b-instruct.ggmlv3.Q3_K_S.bin",
        n_ctx= 2048,
        n_gpu_layers = 1000,  # Change this value based on your model and your GPU VRAM pool.
        n_batch = 512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_threads=None,
        top_k= 10000,
        temperature= 0.0,
        max_tokens=2000,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    
    
    task= input("Task:")
    zeroshot = """
[INST] Your task is to write python code to solve a programming problem. Only provide code.
Please wrap your codeanswer using ```:""" + task +""" [/INST]"""
    oneshot = """
[INST] 
Your task is to write python code to solve a programming problem. Only provide code.
You are given one example from which you can infer the structure of the code:

```
        def main():
            print("Hello World!")

        if __name__ == "__main__":
            main()
```
Please wrap your codeanswer using ```:""" + task +""" [/INST]"""
        
    fewshot = """
[INST] 
Your task is to write python code to solve a programming problem. Only provide code.
You are given two examples from which you can infer the structure of the code:
Example 1:
```
        def main():
            print("Hello World!")

        if __name__ == "__main__":
            main()
```
Example 2:
```
def greet():
    print("Welcome to the Python Syntax Tutorial!")

def get_user_input():
    name = input("Enter your name: ")
    return name

def main():
    greet()

    # Variables
    age = 25
    height = 5.8
    is_student = True

    # Output
    print(f"My age is {age} and my height is {height} feet.")
    print(f"Am I a student? {is_student}")

    # Conditional Statement
    if is_student:
        print("I am a student.")
    else:
        print("I am not a student.")

    # Loop
    for i in range(3):
        print(f"Loop iteration {i + 1}")

    # Function
    def multiply(x, y):
        return x * y

    result = multiply(3, 4)
    print(f"Multiplication result: {result}")

    # User Input and Output
    name = get_user_input()
    print(f"Hello, {name}!")

if __name__ == "__main__":
    main()
```
Please wrap your codeanswer using ```:""" + task +""" [/INST]

Here is the Python code, i came up with:
```
"""
    
    #llm(zeroshot)
    #llm(oneshot)
    print(fewshot)
    llm(fewshot)
    

if __name__ == "__main__":
    main()


