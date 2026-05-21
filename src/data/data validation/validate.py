import os

class DataValidator:

    def __init__(self, num_features=34):
        self.num_features = num_features
        self.expected_cols = num_features + 1

    def validate_file(self, input_path, output_path):

        valid_lines = []

        invalid_count = 0
        duplicate_count = 0
        total_count = 0

        seen = set()

        with open(input_path, "r", encoding="utf-8") as f:

            for line_num, line in enumerate(f, start=1):

                total_count += 1

                line = line.strip()

                #  Empty line 
                if not line:
                    invalid_count += 1
                    continue

                #  Duplicate 
                if line in seen:
                    duplicate_count += 1
                    continue

                seen.add(line)

                values = line.split('\t')

                #  Wrong column count 
                if len(values) != self.expected_cols:
                    invalid_count += 1
                    continue

                #  Validate label 
                label = values[0]

                if label not in ["0", "1"]:
                    invalid_count += 1
                    continue

                #  Validate features 
                features = values[1:]

                has_empty = any(v.strip() == "" for v in features)

                if has_empty:
                    invalid_count += 1
                    continue

                valid_lines.append(line)

        #  Save cleaned data 
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")

        #  Report 
        print("\n VALIDATION REPORT")
        print(f"Input file       : {input_path}")
        print(f"Total samples    : {total_count}")
        print(f"Valid samples    : {len(valid_lines)}")
        print(f"Invalid samples  : {invalid_count}")
        print(f"Duplicate samples: {duplicate_count}")
        print(f"Output file      : {output_path}")
