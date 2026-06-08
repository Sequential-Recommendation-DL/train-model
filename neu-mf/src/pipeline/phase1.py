from src.data.load_data import load_all
from src.data.preprocess import validate, clean

def loading_and_clean_data(mi):
    # loading data
    print("\n[1/6] Loading & cleaning data...")
    df = load_all()

    #print(f"\n - Electronics samples:")
    #print(df.head(2))

    #print(f"\n - Musical instrument samples:")
    #print(df.tail(2))

    print(f"\n - Raw rows: {len(df):,}")

    # clean data
    df = validate(df)
    df = clean(df, mi)
    print(f"After filter: {len(df):,} rows")

    return df
    
