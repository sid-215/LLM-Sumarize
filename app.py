from flask import Flask, request, jsonify 
from TestLLM import get_summary  # Import the function that handles the model in your converted script
from Cleanoutput import CleanedSummary

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Get inputs from the POST request
        data = request.json
        
        # Ensure the key names match the ones you are sending in the POST request
        topic_of_interest = data.get("topic_of_interest")
        timeframe = data.get("timeframe")  # Ensure this matches the key you're sending in the POST request
        
        if not timeframe or not topic_of_interest:
            return jsonify({"error": "Both 'timeframe in weeks' and 'topic_of_interest' are required."}), 400
        
        # Call your model function (this is where your LLM code resides)
        summary = get_summary(topic_of_interest, timeframe)

        cleaned_summary = CleanedSummary(summary=summary).summary

        
        # Return the result as JSON
        return jsonify({"summary": cleaned_summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
