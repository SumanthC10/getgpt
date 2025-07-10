# Save this file as: cli_chat.py
# To run it, use the command: python cli_chat.py
#
# Before running, make sure you have the required libraries installed:
# pip install torch transformers accelerate python-dotenv
#
# And that you have logged in via the terminal:
# huggingface-cli login

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    """
    Loads the TxGemma model and tokenizer from Hugging Face.
    This function assumes you have already logged in via `huggingface-cli login`.
    """
    model_id = "google/txgemma-9b-chat"
    
    print(f"Loading model '{model_id}'... (This might take a moment on first run)")
    
    # Determine the best torch data type for performance on your machine
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # The library will automatically use your saved token from the CLI login
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", # Automatically select the device (mps for Mac, cuda for NVIDIA)
        torch_dtype=torch_dtype,
    )
    print("Model loaded successfully!")
    return tokenizer, model

def main():
    """
    Main function to run the CLI chat loop.
    """
    tokenizer, model = load_model()
    device = model.device
    
    # Store chat history in a simple list
    chat_history = []
    
    print("\n--- Chat with TxGemma ---")
    print("Type 'quit' or 'exit' to end the conversation.")
    
    while True:
        # Get user input from the command line
        prompt = input("\nYou: ")
        
        if prompt.lower() in ["quit", "exit"]:
            print("Exiting chat. Goodbye!")
            break
            
        # Add user's message to the chat history
        chat_history.append({"role": "user", "content": prompt})

        # --- Prepare Model Input ---
        # Format the chat history into the required chat template
        chat_for_model = tokenizer.apply_chat_template(
            chat_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(chat_for_model, return_tensors="pt").to(device)

        # --- Generate Response ---
        print("\nTxGemma: ", end="", flush=True)
        try:
            outputs = model.generate(**inputs, max_new_tokens=512)
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # The response includes the prompt, so we need to extract only the assistant's part.
            model_turn_start = response_text.rfind('<start_of_turn>model\n')
            if model_turn_start != -1:
                full_response = response_text[model_turn_start + len('<start_of_turn>model\n'):].strip()
            else:
                full_response = "Could not parse the model's response."

            print(full_response)
            
            # Add the assistant's response to the chat history
            chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(error_message)
            # Add error to history so the model has context if the user asks about it
            chat_history.append({"role": "assistant", "content": error_message})

if __name__ == '__main__':
    main()
