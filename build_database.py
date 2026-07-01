import pandas as pd
import time
import traceback
from data_loader import fetch_real_season_data

def build_full_nba_archive():
    """Loops through every NBA season from 1950 to 2026 and saves to CSV."""
    all_seasons = []
    
    start_year = 1950
    end_year = 2026
    
    print(f"🚀 Initializing complete NBA historical scrape with PER Support Metrics ({start_year} - {end_year})...")
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year} season... ", end="", flush=True)
        try:
            df_season = fetch_real_season_data(year)
            if not df_season.empty:
                all_seasons.append(df_season)
                print("Success! ✅")
            else:
                print("Empty/Skipped ⚠️")
        except Exception as e:
            print(f"Failed ❌ (Error: {e})")
            traceback.print_exc()
            
        # 1.5-second courtesy pause to protect your IP from rate limits
        time.sleep(1.5)
        
    if all_seasons:
        full_df = pd.concat(all_seasons, ignore_index=True)
        output_file = "nba_raw_archive.csv"
        full_df.to_csv(output_file, index=False)
        print(f"\n🎉 Done! Upgraded database saved successfully as '{output_file}' ({len(full_df)} records).")
    else:
        print("\n❌ Extraction failed. No data compiled.")

if __name__ == "__main__":
    build_full_nba_archive()