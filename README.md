
# Global Power Plant Database Analysis

This project provides an interactive analysis of the Global Power Plant Database using Python and Streamlit. It enables users to explore energy generation patterns worldwide and includes predictive modeling capabilities.

## Features

- **Data Overview**: Examine basic statistics and initial data exploration
- **Data Cleaning**: Handle missing values and prepare data for analysis
- **Exploratory Analysis**: Interactive visualizations of global power plant distributions
- **Feature Engineering**: Create derived features for enhanced analysis
- **Model Building**: Predictive modeling for power generation
- **Model Evaluation**: Performance metrics and visualization tools

## Tech Stack

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Matplotlib
- Seaborn

## Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                         # Main Streamlit application
└── global_power_plant_database.csv # Dataset
```

## Features in Detail

### Data Analysis
- Global distribution of power plants
- Capacity and generation analysis
- Fuel type distribution
- Temporal analysis of commissioning dates

### Visualizations
- Interactive maps of power plant locations
- Time series analysis of power generation
- Capacity factor calculations
- Performance metrics visualization

### Machine Learning
- Predictive modeling for power generation
- Feature importance analysis
- Model performance evaluation

## Data Source

The analysis uses the Global Power Plant Database, which contains information about power plants worldwide including their capacity, location, generation, and fuel types.

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
