import os
import uuid
import telebot
from dotenv import load_dotenv

# Load env in case geochat_url/backend_url are needed
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Make sure TMP_DIR exists
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# We import run_agent from the agent logic
from agent import run_agent

BOT_TOKEN = "8976976840:AAG_Unje01jErnNgAQpzyB25_ZX05lWHmcM"
bot = telebot.TeleBot(BOT_TOKEN)

# Session store per chat ID to handle history and images
# SESSIONS[chat_id] = {"history": [], "images": []}
SESSIONS = {}

def get_session(chat_id):
    if chat_id not in SESSIONS:
        SESSIONS[chat_id] = {"history": [], "images": []}
    return SESSIONS[chat_id]

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_text = (
        "Welcome to the SatQuery Agent.\n\n"
        "This agent is designed to assist with the analysis of remote sensing and satellite imagery. "
        "You can upload up to two images (Optical, SAR, or multispectral) and query the agent to perform analysis.\n\n"
        "Supported Capabilities:\n"
        "- Visual Question Answering: Identify objects or characteristics in a single image.\n"
        "- Change Analysis: Upload a bi-temporal image pair to detect and outline surface changes over time.\n"
        "- Grounding: Locate and outline specific objects or areas mentioned in your query.\n"
        "- Cross-Modal Fusion: Combine co-registered Optical and SAR images for joint identification.\n\n"
        "Available Commands:\n"
        "/start - Initialize the bot.\n"
        "/help - Display this help section.\n"
        "/clear - Purge the current session, resetting conversation history and uploaded images."
    )
    bot.reply_to(message, help_text)

@bot.message_handler(commands=['clear'])
def clear_session(message):
    chat_id = message.chat.id
    if chat_id in SESSIONS:
        SESSIONS[chat_id] = {"history": [], "images": []}
    bot.reply_to(message, "Session cleared. Image paths and history have been reset.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id
    session = get_session(chat_id)
    
    # Get the highest resolution photo
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    # Save file
    file_extension = file_info.file_path.split('.')[-1] if '.' in file_info.file_path else 'jpg'
    unique_filename = f"{uuid.uuid4().hex[:8]}.{file_extension}"
    file_path = os.path.join(TMP_DIR, unique_filename)
    
    with open(file_path, 'wb') as new_file:
        new_file.write(downloaded_file)
        
    session["images"].append(file_path)
    # Keep only the latest 2 images
    session["images"] = session["images"][-2:]
    
    bot.reply_to(message, f"Image received. You have {len(session['images'])} image(s) loaded.")

@bot.message_handler(func=lambda message: True)
def handle_query(message):
    chat_id = message.chat.id
    session = get_session(chat_id)
    query = message.text
    
    # Send a processing message
    processing_msg = bot.reply_to(message, "Processing your query...")
    
    try:
        response = run_agent(query, image_paths=session["images"], history=session["history"])
        
        # Extract analysis
        agent_text = ""
        img_url = None
        
        if isinstance(response, dict):
            if "analysis" in response:
                if isinstance(response["analysis"], dict):
                    agent_text = response["analysis"].get("summary", "")
                else:
                    agent_text = str(response["analysis"])
            else:
                agent_text = str(response)
                
            if "img" in response and response["img"]:
                img_url = response["img"]
        else:
            agent_text = str(response)
            
        # Update history
        session["history"].append({"user": query, "agent": agent_text})
        
        # Send text back
        if agent_text:
            bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text=agent_text)
        else:
            bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text="No textual response.")
            
        # If there's an image in the response, figure out local path and send it
        if img_url:
            # The img_url is typically like: http://localhost:8000/outputs/1234_annotated.png
            # We can extract the filename from the URL and read it from TMP_DIR
            filename = img_url.split("/")[-1]
            local_img_path = os.path.join(TMP_DIR, filename)
            
            if os.path.exists(local_img_path):
                with open(local_img_path, 'rb') as img_file:
                    bot.send_photo(chat_id, img_file)
            else:
                bot.send_message(chat_id, f"Output image generated but could not be found locally: {filename}")
                
    except Exception as e:
        bot.edit_message_text(chat_id=chat_id, message_id=processing_msg.message_id, text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    print("Starting Telegram Bot...")
    bot.infinity_polling()
