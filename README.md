# Conversational Chatbot with LangChain and Gemini

## Introduction
This project is a conversational chatbot built using **LangChain** and **Google Gemini**. It answers user queries based on uploaded documents, provides a conversational form to collect user details when asked to "call me", and allows users to book appointments with date validation. The chatbot can extract and validate user inputs such as email, phone number, and appointment dates.

## Features
- **Document-Based Q&A**: Users can upload PDF documents and ask queries based on the content of the document.
- **Call Me Functionality**: When the user requests a callback by saying "call me", a form is displayed to collect contact details.
- **Appointment Booking**: Users can book appointments by specifying a date in natural language (e.g., "next Monday"). The date is extracted, validated, and confirmed.
- **Validation**: Validation is integrated for email addresses, phone numbers, and dates to ensure proper data collection.
- **Conversational Form**: Smooth interaction with forms to gather user inputs.

## Screenshots

### 1. User Uploads a File and Asks a Query
When the user uploads a document and asks a question based on its content, the chatbot retrieves the relevant information to answer the query.

![Document Query](https://github.com/GarimaPaudel/Chatbot/blob/main/screenshots/Screenshot%201.png)

### 2. User Says "Call Me"
When the user types "call me", the chatbot prompts a conversational form to collect the user's name, phone number, and email.

![Call Me Form](https://github.com/GarimaPaudel/Chatbot/blob/main/screenshots/Screenshot%202.png)

### 3. User Tries to Book an Appointment
When the user mentions booking an appointment with a date (e.g., "Book an appointment for next Monday"), the chatbot extracts and validates the date, confirming the booking.

![Appointment Booking](https://github.com/GarimaPaudel/Chatbot/blob/main/screenshots/Screenshot%203.png)

## Getting Started
### Prerequisites
- Python 3.x
- Install the required dependencies using `pip install -r requirements.txt`

### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/GarimaPaudel/Chatbot.git

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Set up environment variables:
- Create a .env file with your API keys for LangChain and Google Gemini.
GOOGLE_API_KEY ="ENTERYOURKEYHERE"
LANGCHAIN_API_KEY = "ENTERYOURKEYHERE"

4. Run the application:
  ```bash
   streamlit run app.py
    
