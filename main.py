import difflib
import os
import easyocr
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import Levenshtein as lev
from datetime import datetime, date
from jiwer import wer, cer, wil, wip, mer


def clean_record(record):
    # Remove columns with only NaN or empty values
    cleaned_record = {k: v for k, v in record.items() if pd.notna(v) and v != ""}
    return cleaned_record


def format_date(date_str):
    # Check for NaN or NaT
    if pd.isna(date_str):
        return ""  # or return 'NaN'/'NaT' as a string if you prefer
    try:
        # Try to parse the date and format it as 'YYYY-MM-DD'
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        # If parsing fails, return the original string
        return date_str


class IDReader:
    def __init__(self):
        self.data_path = "data"
        self.results = pd.DataFrame()
        self.easyocr_reader = None

    def initiate_ocr_reader(self, languages: list):
        self.easyocr_reader = easyocr.Reader(lang_list=languages)

    def read_jpg_file(self, file: str):
        try:
            path = os.path.join(self.data_path, file)
            image = Image.open(path)
            results = self.easyocr_reader.readtext(image=image)

            if len(results) < 27:
                print(f"Insufficient OCR results for {file}")
                return

            # Extract text from results
            text_results = [text for _, text, _ in results]

            # Words to match
            words_to_be_matched = [
                "primer appelido",
                "segundo appelido",
                "nombre",
                "sexo",
                "nacionalidad",
                "fecha de nacimiento",
                "idesp",
                "valido hasta",
            ]

            # Match words using similarity
            matched = {
                word: max(
                    text_results,
                    key=lambda result: difflib.SequenceMatcher(
                        a=word.lower(), b=result.lower()
                    ).ratio(),
                )
                for word in words_to_be_matched
            }

            record = {"associated_document": file}

            for word, match in matched.items():
                if match in text_results:
                    match_index = text_results.index(match)

                    # Handle specific cases like dates and other fields
                    if word == "fecha de nacimiento" or word == "valido hasta":
                        delta = 1 if word == "fecha de nacimiento" else 0
                        try:
                            day, month, year = map(
                                int,
                                text_results[
                                    match_index + 1 + delta : match_index + 4 + delta
                                ],
                            )
                            associated_text = date(day=day, month=month, year=year)
                        except Exception:
                            associated_text = None
                    else:
                        distance = 2 if word in ["nacionalidad", "sexo", "idesp"] else 1
                        associated_text = text_results[match_index + distance].upper()

                    # Populate record with associated text
                    if word == "primer appelido":
                        record["name"] = associated_text
                    elif word == "segundo appelido":
                        record["name"] = record["name"] + " " + associated_text
                    elif word == "nombre":
                        record["surname"] = associated_text
                    elif word == "sexo":
                        record["gender"] = associated_text
                    elif word == "nacionalidad":
                        record["nationality"] = associated_text
                    elif word == "fecha de nacimiento":
                        record["birthdate"] = associated_text
                    elif word == "idesp":
                        record["country_id"] = associated_text
                    elif word == "valido hasta":
                        record["validity_end_date"] = associated_text

            # Add record to results DataFrame
            record = clean_record(record)
            if record:
                self.results = pd.concat(
                    [self.results, pd.DataFrame([record])], ignore_index=True
                )
            else:
                print("Record is empty ot contains only NaN values, skipping concat.")

        except Exception as e:
            print(f"Error processing {file}: {e}")

        # try:
        #     path = os.path.join(self.data_path, file)
        #     image = Image.open(path)
        #     results = self.easyocr_reader.readtext(image=image)
        #
        #     if len(results) < 27:  # Ensure there are enough elements in results
        #         print(f"Insufficient OCR results for {file}")
        #         return
        #
        #     # Extract text based on index positions, adjust these if needed
        #     text_results = [text for bbox, text, prob in results]
        #     # print(text_results)
        #     words_to_be_matched = [
        #         "primer appelido",
        #         "segundo appelido",
        #         "nombre",
        #         "sexo",
        #         "nacionalidad",
        #         "fecha de nacimiento",
        #         "idesp",
        #         "valido hasta",
        #     ]
        #     matched = {}
        #     for word in words_to_be_matched:
        #         highest_similarity = 0
        #         most_accurate_match = None
        #         for result in text_results:
        #             similarity = difflib.SequenceMatcher(
        #                 isjunk=None, a=word.lower(), b=result.lower()
        #             ).ratio()
        #             if similarity > highest_similarity:
        #                 highest_similarity = similarity
        #                 most_accurate_match = result
        #         matched[word] = most_accurate_match
        #     record = {
        #         "associated_document": file,
        #     }
        #     # Find the associated text in text_results (the result right after the match)
        #     for word, match in matched.items():
        #         if match in text_results:
        #             match_index = text_results.index(
        #                 match
        #             )  # Get the index of the match
        #             if match_index + 1 < len(
        #                 text_results
        #             ):  # Ensure we don't go out of bounds
        #                 if word in ["fecha de nacimiento", "valido hasta"]:
        #                     delta = 0
        #                     if word == "fecha de nacimiento":
        #                         delta += 1
        #                     day = text_results[match_index + 1 + delta]
        #                     month = text_results[match_index + 2 + delta]
        #                     year = text_results[match_index + 3 + delta]
        #                     try:
        #                         associated_text = datetime(
        #                             day=int(day), month=int(month), year=int(year)
        #                         )
        #                     except Exception as e:
        #                         associated_text = ""
        #                         print(f"\n{e}")
        #                 else:
        #                     if word in ["nacionalidad", "sexo", "idesp"]:
        #                         distance = 2
        #                     else:
        #                         distance = 1
        #                     associated_text = text_results[
        #                         match_index + distance
        #                     ].upper()
        #             else:
        #                 associated_text = (
        #                     None  # Handle the case where there's no "next" item
        #                 )
        #             # You can add to the record here based on the word
        #             if (
        #                 word == "primer appelido"
        #             ):  # Modify based on your words_to_be_matched keys
        #                 record["name"] = associated_text.upper()
        #             elif word == "segundo appelido":
        #                 record["name"] = record["name"] + " " + associated_text.upper()
        #             elif word == "nombre":
        #                 record["surname"] = associated_text.upper()
        #             elif word == "sexo":
        #                 record["gender"] = associated_text.upper()
        #             elif word == "nacionalidad":
        #                 record["nationality"] = associated_text.upper()
        #             elif word == "fecha de nacimiento":
        #                 record["birthdate"] = associated_text
        #             elif word == "idesp":
        #                 record["country_id"] = associated_text.upper()
        #             elif word == "valido hasta":
        #                 record["validity_end_date"] = associated_text
        #     df = pd.DataFrame(
        #         data=[record]
        #     )  # Wrap the record in a list to create a DataFrame
        #     self.results = pd.concat(objs=[self.results, df], ignore_index=True)
        #     # print(f"Processed {file}")
        # except Exception as e:
        #     print(f"Error processing {file}: {e}")

    def compute_levenshtein_distance(
        self, ocr_results: pd.DataFrame, reference: pd.DataFrame
    ):

        # Compute Levenshtein distance for each cell
        lev_distances = ocr_results.apply(
            lambda row: pd.Series(
                [lev.distance(a, b) for a, b in zip(row, ocr_results.loc[row.name])],
                index=row.index,
            ),
            axis=1,
        )

        # Calculate the total distance per row and per column
        total_distance_per_row = lev_distances.sum(axis=1)
        total_distance_per_column = lev_distances.sum(axis=0)

        print(f"Levenshtein distances (per row):\n{total_distance_per_row}")
        print(f"Levenshtein distances (per column):\n{total_distance_per_column}")

        # Find rows where the total distance is zero (perfect match)
        exact_matches = total_distance_per_row[total_distance_per_row == 0]

        print(f"Number of exact matching rows: {len(exact_matches)}")

    def compute_ocr_accuracy(self, reference: pd.DataFrame):
        # Compare the source data with the OCR results
        diff = reference != self.results

        # Identify identical rows
        identical_rows = ~diff.any(axis=1)

        # Calculate the total Word Error Rate
        ocr_accuracy = identical_rows.mean() * 100

        # Calculate the Word Error Rate per column
        ocr_accuracy_per_columns = ((~diff).mean(axis=0) * 100).rename("OCR Accuracy")

        # Convert the Series to DataFrame for better handling
        ocr_accuracy_df = pd.DataFrame(
            data=ocr_accuracy_per_columns, columns=["OCR Accuracy"]
        )

        # Print results
        print(f"OCR Accuracy: {ocr_accuracy:.2f}%")
        print(f"\nOCR Accuracy per Column:\n{ocr_accuracy_df}")

    def wer_cell(self, ref, cell):
        ref = str(ref)
        cell = str(cell)
        return wer(ref, cell)

    def cer_cell(self, ref, cell):
        ref = str(ref)
        cell = str(cell)
        return cer(ref, cell)

    def compute_levenshtein(self, reference, hypothesis):
        m = len(reference)
        n = len(hypothesis)

        matrix = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            matrix[i][0] = i
        for j in range(n + 1):
            matrix[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i - j][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost,
                )
        levenshtein_distance = matrix[m][n]
        return levenshtein_distance

    def compute_cer(self, reference: str, hypothesis: str):
        cer = self.compute_levenshtein(reference, hypothesis) / len(reference) if len(reference) > 0 else 0
        return cer

    def compute_word_error_rate_and_character_error_rate(
        self, ocr_results: pd.DataFrame, reference: pd.DataFrame
    ):
        wer_matrix = pd.DataFrame(index=ocr_results.index, columns=ocr_results.columns)
        cer_matrix = pd.DataFrame(index=ocr_results.index, columns=ocr_results.columns)

        for i in range(ocr_results.shape[0]):
            for j in range(ocr_results.shape[1]):
                wer_matrix.iat[i, j] = self.wer_cell(
                    reference.iat[i, j], ocr_results.iat[i, j]
                )
                cer_matrix.iat[i, j] = self.cer_cell(
                    reference.iat[i, j], ocr_results.iat[i, j]
                )

        # Calculate the total Word Error Rate per row and per column
        total_wer_per_row = wer_matrix.sum(axis=1)
        total_wer_per_column = wer_matrix.sum(axis=0)

        # Calculate the total Character Error Rate per row and per column
        total_cer_per_row = cer_matrix.sum(axis=1)
        total_cer_per_column = cer_matrix.sum(axis=0)

        print(f"Word Error Rate (per row):\n{total_wer_per_row}")
        print(f"Word Error Rate (per column):\n{total_wer_per_column}")

        print(f"Character Error Rate (per row):\n{total_cer_per_row}")
        print(f"Character Error Rate (per column):\n{total_cer_per_column}")

        return wer_matrix, cer_matrix

    def evaluate_ocr_results(self, reference_file_path: str):
        # Read the Excel file into a DataFrame
        with open(file=reference_file_path, mode="rb") as f:
            byte_content = f.read()
        byte_io_object = BytesIO(initial_bytes=byte_content)
        data = pd.read_excel(io=byte_io_object)

        self.compute_ocr_accuracy(reference=data)

        # Check if the dimensions match between source data and OCR results
        if data.shape != self.results.shape:
            print("Source data and OCR results have different dimensions.")
            return

        # Ensure all data is converted to strings
        source_data_str = data.astype(str)
        ocr_results_str = self.results.astype(str)

        source_data_str["birthdate"] = source_data_str["birthdate"].apply(format_date)
        source_data_str["validity_end_date"] = source_data_str[
            "validity_end_date"
        ].apply(format_date)

        ocr_results_str["birthdate"] = ocr_results_str["birthdate"].apply(format_date)
        ocr_results_str["validity_end_date"] = ocr_results_str[
            "validity_end_date"
        ].apply(format_date)

        self.compute_levenshtein_distance(
            ocr_results=ocr_results_str, reference=source_data_str
        )
        self.compute_word_error_rate_and_character_error_rate(
            ocr_results=ocr_results_str, reference=source_data_str
        )


if __name__ == "__main__":
    reader = IDReader()
    reader.initiate_ocr_reader(languages=["es"])
    for file in tqdm(os.listdir(path=reader.data_path), desc="Processing files"):
        if file.endswith(".jpg"):
            reader.read_jpg_file(file=file)
        else:
            print(f"File {file} is not a jpg.")
    reader.evaluate_ocr_results(
        reference_file_path="ref/data_esp_id.xlsx"
    )
    print("")
