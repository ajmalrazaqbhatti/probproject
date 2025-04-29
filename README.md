# Insurance Data Analysis Project

## Setup Instructions

### Option 1: Using setup script (recommended)

1. Make the setup script executable:

   ```
   chmod +x setup.sh
   ```

2. Run the setup script:

   ```
   ./setup.sh
   ```

3. Activate the virtual environment:

   ```
   source venv/bin/activate
   ```

4. Run the application:

   ```
   streamlit run main.py
   ```

   Or use the run script:

   ```
   ./run.sh
   ```

### Option 2: Manual setup

1. Create a virtual environment:

   ```
   python3 -m venv venv
   ```

2. Activate the virtual environment:

   ```
   source venv/bin/activate
   ```

3. Install requirements:

   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run main.py
   ```

## Project Structure

- `main.py`: Main Streamlit application
- `insurance.csv`: Dataset for analysis
- `requirements.txt`: Python dependencies

## Development

- Make sure to activate the virtual environment before development
- Add new dependencies to requirements.txt as needed
