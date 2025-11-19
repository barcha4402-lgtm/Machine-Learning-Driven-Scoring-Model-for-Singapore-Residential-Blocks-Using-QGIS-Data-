"""
Predict and Score System
Predict building metrics and calculate comprehensive scores
"""

import sys
import argparse
from pathlib import Path
from model_saver_and_predictor import ModelSaverAndPredictor
from scoring_system import ScoringSystem
import pandas as pd


def predict_and_score(input_file, output_file='predictions_with_scores.xlsx'):
    """
    Predict building metrics and calculate scores
    
    Parameters:
    -----------
    input_file : str
        Path to input Excel file
    output_file : str
        Path to output Excel file (default: 'predictions_with_scores.xlsx')
    
    Returns:
    --------
    DataFrame
        Results with predictions and scores
    """
    print("="*80)
    print("Predict and Score System")
    print("="*80)
    
    # 1. Check input file
    print(f"\n[Step 1] Checking input file...")
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"  ERROR: File not found: {input_file}")
        return None
    
    if not input_path.suffix.lower() in ['.xlsx', '.xls']:
        print(f"  ERROR: File must be Excel format (.xlsx or .xls)")
        return None
    
    print(f"  OK: File found: {input_file}")
    
    # 2. Load model
    print(f"\n[Step 2] Loading model...")
    predictor = ModelSaverAndPredictor()
    
    if not predictor.load_from_files():
        print("  ERROR: Cannot load model file saved_models.pkl")
        print("  Please ensure the model file exists")
        return None
    
    print("  OK: Model loaded successfully")
    print(f"  Loaded {len(predictor.models)} models:")
    for target_name in predictor.models.keys():
        print(f"    - {target_name}")
    
    # 3. Load input data
    print(f"\n[Step 3] Loading input data...")
    try:
        input_data = pd.read_excel(input_file)
        print(f"  OK: Loaded {len(input_data)} rows, {len(input_data.columns)} columns")
    except Exception as e:
        print(f"  ERROR: Failed to load file: {e}")
        return None
    
    # 4. Predict
    print(f"\n[Step 4] Making predictions...")
    predictions_df = predictor.predict_all(input_data)
    
    if predictions_df is None:
        print("  ERROR: Prediction failed")
        return None
    
    print("  OK: Prediction completed")
    print(f"  Predicted {len(predictions_df)} samples")
    
    # Show prediction summary
    print(f"\nPrediction Summary:")
    for col in predictions_df.columns:
        print(f"  {col}:")
        print(f"    Range: [{predictions_df[col].min():.4f}, {predictions_df[col].max():.4f}]")
        print(f"    Mean: {predictions_df[col].mean():.4f}")
    
    # 5. Calculate scores
    print(f"\n[Step 5] Calculating scores...")
    scoring_system = ScoringSystem()
    scores_df = scoring_system.score_predictions(predictions_df, input_data)
    
    # Add score level
    scores_df['score_level'] = scores_df['total_score'].apply(
        scoring_system.get_score_description
    )
    
    print("  OK: Scoring completed")
    
    # Show score summary
    print(f"\nScore Summary:")
    print(f"  Average Total Score: {scores_df['total_score'].mean():.2f}")
    print(f"  Score Range: [{scores_df['total_score'].min():.2f}, {scores_df['total_score'].max():.2f}]")
    print(f"\nAverage Scores by Category:")
    print(f"  Accessibility: {scores_df['accessibility_score'].mean():.2f}")
    print(f"  Plot Ratio: {scores_df['plot_ratio_score'].mean():.2f}")
    print(f"  Greenery Ratio: {scores_df['greenery_ratio_score'].mean():.2f}")
    print(f"  Water Ratio: {scores_df['water_ratio_score'].mean():.2f}")
    print(f"  Orientation: {scores_df['orientation_score'].mean():.2f}")
    
    # Score distribution
    print(f"\nScore Distribution:")
    score_levels = scores_df['score_level'].value_counts()
    for level, count in score_levels.items():
        print(f"  {level}: {count} ({count/len(scores_df)*100:.1f}%)")
    
    # 6. Merge results
    print(f"\n[Step 6] Merging results...")
    result_df = pd.concat([input_data, predictions_df, scores_df], axis=1)
    
    # 7. Save results
    print(f"\n[Step 7] Saving results: {output_file}")
    try:
        result_df.to_excel(output_file, index=False)
        print("  OK: Saved successfully")
    except Exception as e:
        print(f"  ERROR: Failed to save: {e}")
        return None
    
    # 8. Show top 10 highest scores
    print(f"\nTop 10 Highest Scores:")
    top_10 = result_df.nlargest(10, 'total_score')[
        ['plot_ratio', 'accessibility', 'greenery_ratio', 'water_ratio', 
         'total_score', 'score_level']
    ]
    print(top_10.to_string())
    
    print("\n" + "="*80)
    print("Prediction and Scoring Completed!")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print("\nOutput includes:")
    print("  - Original input data")
    print("  - Predictions (plot_ratio, accessibility, greenery_ratio, water_ratio)")
    print("  - Individual scores (accessibility_score, plot_ratio_score, etc.)")
    print("  - Total score (total_score)")
    print("  - Score level (score_level: Excellent/Good/Medium/Fair/Poor)")
    
    return result_df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Predict building metrics and calculate comprehensive scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_and_score.py input.xlsx
  python predict_and_score.py input.xlsx -o output.xlsx
  python predict_and_score.py "C:/data/my_data.xlsx"
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input Excel file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='predictions_with_scores.xlsx',
        help='Path to output Excel file (default: predictions_with_scores.xlsx)'
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path('saved_models.pkl').exists():
        print("\nERROR: saved_models.pkl not found")
        print("Please train and save the model first")
        sys.exit(1)
    
    # Run prediction and scoring
    result = predict_and_score(
        input_file=args.input_file,
        output_file=args.output
    )
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

