import os
import json
import time
import re
import statistics
import collections 
from datetime import datetime, timezone 
from langdetect import detect, LangDetectException
from ollama import chat

# FOLDER = "C:/Users/Cade/RealFusionProject" 
FOLDER = r"C:\Users\Cade\RealFusionProject" 
OUTPUT_REPORT = os.path.join(FOLDER, "sentiment_report.txt")
OUTPUT_JSON = os.path.join(FOLDER, "all_sentiment_outputs.json") # For intermediate results
MODEL = 'llama3.2:3b'
IRRELEVANT_TOKEN = "IRRELEVANT" # Define a clear token for non-US articles

# --- Configuration ---
BATCH_SIZE = 100 # How many article results to summarize at a time
LLM_CTX_ARTICLE = 4096 # Context window for single article analysis
LLM_CTX_BATCH = 8192   # Context window for batch summary (adjust based on BATCH_SIZE and model limits)
LLM_CTX_FINAL = 16384  # Context window for final summary (adjust based on number of batches and model limits)
DATE_FORMAT = '%Y-%m-%d' 


def parse_date_or_none(d_str):
    if not d_str:
        return None
    try:
        # Handle 'Z' indicating UTC
        if d_str.endswith("Z"):
            d_str = d_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(d_str)
        # Ensure timezone-aware (assume UTC if naive, though fromisoformat usually handles offset)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc) # Make it timezone-aware (UTC)
        return dt
    except (ValueError, TypeError):
        return None

# --- Pass 1: Article Analysis ---
start_analysis = time.time()
print("--- Starting Pass 1: Article Sentiment Analysis ---")

all_sentiment_outputs = []
iterate = 0
total_processed = 0
total_relevant = 0
total_irrelevant = 0
total_failed_lang_detect = 0
total_failed_parse = 0
total_failed_llm_format = 0

print("Counting total articles...")
content_count = 0
log_files = [f for f in os.listdir(FOLDER) if f.endswith(".log")]
log_files = [f for f in log_files if f not in (os.path.basename(OUTPUT_REPORT), os.path.basename(OUTPUT_JSON))]

for file_name in log_files:
    file_path = os.path.join(FOLDER, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            content_count += sum(1 for article in data if "content" in article and article["content"])
    except Exception as e:
        print(f"Warning reading {file_name}: {e}")

print(f"Found approximately {content_count} articles with 'content'.")

for file_name in log_files:
    file_path = os.path.join(FOLDER, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

    articles_in_file = [article for article in data if article.get("content")]
    if not articles_in_file:
        continue

    for article_index, article in enumerate(articles_in_file):
        iterate += 1
        text = article.get("content")
        pub_date_obj = parse_date_or_none(article.get("estimatedPublishedDate"))
        article_id = article.get("id", f"file_{file_name}_index_{article_index}")
        print(f"Processing: Article {iterate}/{content_count} [File: {file_name}, Item: {article_index+1}/{len(articles_in_file)}]")

        if len(text) < 20:
            print(f"  Skipping: Content too short ({len(text)} chars).")
            total_failed_lang_detect += 1
            continue
        try:
            if detect(text[:500]) != 'en':
                print("  Skipping: Non-English detected.")
                total_failed_lang_detect += 1
                continue
        except LangDetectException:
            print("  Skipping: Language detection failed.")
            total_failed_lang_detect += 1
            continue

        # --- US Context Check ---
        if not re.search(r'\b(United States|U\.S\.|US)\b', text, re.IGNORECASE):
            print("  Result: Irrelevant (Not primarily about US)")
            total_irrelevant += 1
            all_sentiment_outputs.append({
                "id": article_id,
                "file": file_name,
                "pub_date_obj": pub_date_obj,
                "is_relevant": False,
                "score": IRRELEVANT_TOKEN,
                "justification": "Article is not primarily about the US."
            })
            continue

        # --- Prompt for structured sentiment and relevance ---
        prompt = (
            f"Analyze the sentiment expressed towards *nuclear fusion* in the following text. The text MUST be primarily and explicitly about the *United States*.\n"
            f"1. First, assess relevance: Is the text primarily and explicitly about the United States? (Yes/No)\n"
            f"2. If No, respond ONLY with: Relevance: No\n"
            f"3. If Yes, respond with the sentiment score towards *nuclear fusion* on a Likert scale from 1 to 5 (1=very negative, 3=neutral/technical, 5=very positive) AND a brief (1-sentence) justification for the score. Format your response EXACTLY like this (with the newline):\n"
            f"Score: [number]\n"
            f"Justification: [Your brief justification here]\n\n"
            f"Text:\n{text[:LLM_CTX_ARTICLE - 600]}" # Truncate text, reserve more space for clearer instructions
        )

        try:
            response_data = chat(
                model=MODEL,
                options={
                    "num_ctx": LLM_CTX_ARTICLE,
                    "temperature": 0.1 # <<< CHANGE: Even lower temp for stricter formatting
                    },
                messages=[{"role": "user", "content": prompt}]
            )
            llm_output = response_data.get('message', {}).get('content', '').strip()

            if not llm_output:
                 print(f"  Error: LLM returned empty response for article {iterate} in {file_name}.")
                 total_failed_llm_format += 1
                 continue

            # --- Parse the structured LLM output ---
            score = None
            justification = None
            is_relevant = False 

            # Use case-insensitive check for robustness
            if llm_output.lower().startswith("relevance: no"):
                is_relevant = False
                total_irrelevant += 1
                print("  Result: Irrelevant (Not primarily about US)")
            else:
                score_match = re.search(r"Score:\s*([1-7])", llm_output, re.IGNORECASE)
                justification_match = re.search(r"Justification:\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)

                if score_match:
                    score = int(score_match.group(1))
                    is_relevant = True 
                    total_relevant += 1
                    if justification_match:
                        # Remove potential leading/trailing whitespace and quote marks if LLM adds them
                        justification = justification_match.group(1).strip().strip('"')
                    else:
                        justification = "[Justification missing or parsing failed]"
                        print("  Warning: Score found, but justification parsing failed.")
                    print(f"  Result: Relevant (US Context), Fusion Sentiment Score={score}")
                else:
                    # LLM failed to follow format for a relevant article
                    print(f"  Warning: LLM response did not match expected score format for relevant article: '{llm_output[:100]}...'")
                    total_failed_llm_format += 1
                    continue # Skip appending this article

            total_processed += 1
            # Store structured results
            all_sentiment_outputs.append({
                "id": article_id,
                "file": file_name,       
                "pub_date_obj": pub_date_obj,
                "is_relevant": is_relevant, # Is about US context
                "score": score, # Sentiment score for Fusion
                "justification": justification
            })

        except Exception as e:
            print(f"Error: LLM call or processing failed for article {iterate} in {file_name}: {e}")
            total_failed_llm_format += 1
            continue

# --- End of Pass 1 ---
end_analysis = time.time()
print("\n--- Pass 1 Summary ---")
print(f"Total .log files scanned: {len(log_files)}")
print(f"Approximate articles found initially: {content_count}")
print(f"Articles processed (attempted analysis): {total_processed}")
print(f"  Relevant (US Context, Scored for Fusion): {total_relevant}")
print(f"  Irrelevant (Not US Context): {total_irrelevant}")
print(f"Articles Skipped Before Analysis:")
print(f"  Skipped (Lang Detect Fail/Short): {total_failed_lang_detect}")
print(f"  Skipped (LLM Format/Error/Empty): {total_failed_llm_format}")
print(f"Pass 1 duration: {end_analysis - start_analysis:.2f} seconds.")


print(f"\nSaving intermediate results to {OUTPUT_JSON}...")
try:
    # <<< CHANGE: Convert datetime objects to ISO strings for JSON serialization
    serializable_outputs = []
    for item in all_sentiment_outputs:
        new_item = item.copy()
        pub_date_obj = new_item.pop("pub_date_obj", None) # Remove the object
        new_item["pub_date"] = pub_date_obj.isoformat() if pub_date_obj else None # Add ISO string
        serializable_outputs.append(new_item)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable_outputs, f, indent=2)
    print("Intermediate results saved.")
except Exception as e:
    print(f"Error: Could not save intermediate results: {e}")


# --- Pass 2: Batch Summarization ---
start_summary = time.time()
print("\n--- Starting Pass 2: Batch Summarization ---")

# Optional: Load results if restarting
# print("Loading intermediate results...")
# try:
#     with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
#         loaded_intermediate_data = json.load(f)
#     all_sentiment_outputs = []
#     for item in loaded_intermediate_data:
#         item["pub_date_obj"] = parse_date_or_none(item.get("pub_date"))
#         all_sentiment_outputs.append(item)
#     print(f"Loaded {len(all_sentiment_outputs)} results and parsed dates.")
# except FileNotFoundError:
#     print("Error: Intermediate JSON file not found. Cannot proceed with summarization.")
#     exit() 
# except Exception as e:
#     print(f"Error loading intermediate JSON: {e}")
#     exit() 


# Filter for relevant articles (US context) AND sort by date
# Use datetime.min or datetime.max to control sorting of None dates
min_datetime = datetime.min.replace(tzinfo=timezone.utc) # Timezone-aware min datetime

relevant_outputs = [item for item in all_sentiment_outputs if item.get("is_relevant")]
relevant_outputs.sort(key=lambda x: x.get("pub_date_obj") or min_datetime)

print(f"Proceeding to summarize {len(relevant_outputs)} relevant articles (US Context, Scored for Fusion).")

# Function to chunk data
def chunk(items, size):
    if size <= 0:
        yield items # Yield all items as one chunk if size is invalid
        return
    for i in range(0, len(items), size):
        yield items[i:i+size]

batch_summaries_data = [] # Store structured batch data + LLM summary

with open(OUTPUT_REPORT, "w", encoding="utf-8") as out_file:
    out_file.write("--- SENTIMENT ANALYSIS REPORT ---\n")
    out_file.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    out_file.write(f"Topic: Sentiment towards Nuclear Fusion within US-related articles\n") # <<< CHANGE: Clarify topic
    out_file.write(f"Model used for analysis: {MODEL}\n")
    out_file.write(f"Total relevant articles analyzed: {len(relevant_outputs)}\n\n")
    out_file.write("--- BATCH SUMMARIES ---\n\n")

    if not relevant_outputs:
        print("No relevant articles found to summarize.")
        out_file.write("No relevant articles found to summarize.\n")
    else:
        for batch_i, batch_items in enumerate(chunk(relevant_outputs, BATCH_SIZE), start=1):
            print(f"Summarizing Batch {batch_i} ({len(batch_items)} relevant articles)...")

            # <<< CHANGE: Extract batch date range more robustly ---
            batch_start_date_obj = None
            batch_end_date_obj = None
            valid_dates_in_batch = [item.get("pub_date_obj") for item in batch_items if item.get("pub_date_obj")]

            if valid_dates_in_batch:
                batch_start_date_obj = min(valid_dates_in_batch)
                batch_end_date_obj = max(valid_dates_in_batch)
                start_date_str = batch_start_date_obj.strftime(DATE_FORMAT)
                end_date_str = batch_end_date_obj.strftime(DATE_FORMAT)
                batch_date_range_str = f"{start_date_str} to {end_date_str}"
            else:
                start_date_str = "Unknown"
                end_date_str = "Unknown"
                batch_date_range_str = "Unknown date range"
            # --- End Date Range Extraction ---

            # Filter items with valid scores *within the batch*
            scored_items = [item for item in batch_items if item.get('score') is not None]
            batch_scores = [item['score'] for item in scored_items]

            if not batch_scores:
                print(f"  Batch {batch_i} ({batch_date_range_str}) has no valid scores. Skipping summary.")
                summary_text = "No valid sentiment scores found in this batch."
                avg_score = None
                freq_dist_str = "N/A"
                stats_package = {
                    "average_score": None, "median_score": None, "mode_score": "N/A", "frequency": {}, "num_scores": 0
                }
                batch_summaries_data.append({
                     "batch_num": batch_i,
                     "num_articles": len(batch_items),
                     "start_date": start_date_str,
                     "end_date": end_date_str,
                     "date_range": batch_date_range_str,
                     **stats_package, # Unpack stats
                     "llm_summary": summary_text
                })
                # <<< CHANGE: Include date range in the "no scores" output
                out_file.write(f"Batch {batch_i} ({batch_date_range_str}): {summary_text}\n\n")
                continue

            # --- Calculate statistics MANUALLY ---
            avg_score = statistics.mean(batch_scores)
            median_score = statistics.median(batch_scores)
            try:
                # Use multimode if you expect multiple modes, otherwise mode raises error
                modes = statistics.multimode(batch_scores)
                mode_score = ", ".join(map(str, sorted(modes))) if modes else "N/A"
            except statistics.StatisticsError:
                 # This might happen if all values are unique, though multimode should handle it.
                 # Fallback to simple mode just in case.
                 try:
                     mode_score = str(statistics.mode(batch_scores))
                 except statistics.StatisticsError:
                      mode_score = "N/A (no unique mode)"

            freq_dist = collections.Counter(batch_scores)
            freq_dist_str = ", ".join([f"Score {s}: {c}" for s, c in sorted(freq_dist.items())])

            # Package stats for storage and prompt
            stats_package = {
                "average_score": avg_score,
                "median_score": median_score,
                "mode_score": mode_score,
                "frequency": dict(freq_dist),
                "num_scores": len(batch_scores)
            }

            # --- Prepare text for LLM context (using scored items) ---
            context_lines = []
            for item in scored_items: # Iterate through items that actually have scores
                date_obj = item.get("pub_date_obj")
                date_str = date_obj.strftime(DATE_FORMAT) if date_obj else "Unknown"

                score = item.get('score', '?') # Should have a score here
                just = item.get('justification', 'N/A').replace("\n", " ").strip()[:150] # Limit length
                context_lines.append(f"- Date: {date_str}, Score: {score}, Justification: {just}")

            combined_context = "\n".join(context_lines)

            # --- Prompt for Batch Summary ---
            prompt = (
                f"Below are {stats_package['num_scores']} sentiment results towards *nuclear fusion* from news articles related to the US, covering the period {batch_date_range_str}.\n\n"
                f"--- Calculated Statistics for this Batch ---\n"
                f"Date Range: {batch_date_range_str}\n"
                f"Average Score (1-7): {avg_score:.2f}\n"
                f"Median Score: {median_score}\n"
                f"Mode Score(s): {mode_score}\n"
                f"Score Frequency: {freq_dist_str}\n"
                f"Number of Scores: {stats_package['num_scores']}\n\n"
                f"--- Individual Results (Date, Score & Justification) ---\n"
                f"{combined_context}\n\n"
                f"--- Task ---\n"
                f"1. Based ONLY on the justifications provided above and the calculated statistics, write a concise summary (2-3 sentences) describing the dominant sentiment themes towards *nuclear fusion* within this batch.\n"
                f"2. **Analyze Temporal Trend:** Look at the sequence of dates and scores/justifications. Briefly comment (1-2 sentences) if you observe any noticeable trend in sentiment (e.g., becoming more positive, more negative, mixed, staying consistent) during the period {batch_date_range_str}. If no clear trend is apparent, state that.\n"
                f"Combine these into a single response. Do not recalculate statistics."
            )


            try:
                response = chat(
                    model=MODEL,
                    options={
                        "num_ctx": LLM_CTX_BATCH,
                        "temperature": 0.5 # Allow some creativity in summarization/trend description
                        },
                    messages=[{"role": "user", "content": prompt}]
                )
                summary_text = response.get('message', {}).get('content', '').strip()
                if not summary_text:
                     summary_text = "[LLM returned empty summary for Batch {batch_i}]"
                     print(f"  Warning: LLM returned empty summary for Batch {batch_i}")


                print(f"  Batch {batch_i}: Avg Score={avg_score:.2f}, Date Range={batch_date_range_str}, Summary received.")

            except Exception as e:
                print(f"Error: LLM call failed for batch {batch_i} summary: {e}")
                summary_text = f"[LLM Summary failed for Batch {batch_i}: {e}]"

            # Store batch results
            batch_summaries_data.append({
                "batch_num": batch_i,
                "num_articles": len(batch_items), # Total relevant articles in batch
                "start_date": start_date_str,
                "end_date": end_date_str,
                "date_range": batch_date_range_str,
                **stats_package, # Unpack calculated stats
                "llm_summary": summary_text # Includes themes and trend analysis
            })

            # Write batch summary to report
            # <<< CHANGE: Include date range in header
            out_file.write(f"Batch {batch_i} Summary ({stats_package['num_scores']} scored articles, Date Range: {batch_date_range_str}, Avg Score: {avg_score:.2f}):\n")
            out_file.write(f"  Stats: {freq_dist_str}\n")
            out_file.write(f"  LLM Summary & Trend Analysis:\n{summary_text}\n\n")


    # --- Pass 3: Final Summary ---
    print("\n--- Starting Pass 3: Final Report Generation ---")

    if not relevant_outputs:
        out_file.write("\n--- FINAL SUMMARY ---\n\n")
        out_file.write("No relevant articles were found to generate a final summary.\n")
        print("Skipping final summary as no relevant articles were processed.")

    else:
        # --- Calculate OVERALL statistics ---
        # Use only scores from relevant items
        all_scored_items = [item for item in relevant_outputs if item.get('score') is not None]
        all_scores = [item['score'] for item in all_scored_items]

        if not all_scores:
             out_file.write("\n--- FINAL SUMMARY ---\n\n")
             out_file.write("No valid scores found across all relevant articles.\n")
             print("Skipping final summary as no valid scores were found.")
        else:
            overall_avg_score = statistics.mean(all_scores)
            overall_median_score = statistics.median(all_scores)
            try:
                overall_modes = statistics.multimode(all_scores)
                overall_mode_score = ", ".join(map(str, sorted(overall_modes))) if overall_modes else "N/A"
            except statistics.StatisticsError:
                 try:
                     overall_mode_score = str(statistics.mode(all_scores))
                 except statistics.StatisticsError:
                      overall_mode_score = "N/A (no unique mode)"


            overall_freq_dist = collections.Counter(all_scores)
            overall_freq_dist_str = ", ".join([f"Score {s}: {c}" for s, c in sorted(overall_freq_dist.items())])
            total_score_count = len(all_scores)

            # <<< CHANGE: Determine overall date range ---
            all_valid_dates = [item.get("pub_date_obj") for item in all_scored_items if item.get("pub_date_obj")]
            overall_start_date_str = min(all_valid_dates).strftime(DATE_FORMAT) if all_valid_dates else "Unknown"
            overall_end_date_str = max(all_valid_dates).strftime(DATE_FORMAT) if all_valid_dates else "Unknown"
            overall_date_range_str = f"{overall_start_date_str} to {overall_end_date_str}" if all_valid_dates else "Unknown"
            # --- End Overall Date Range ---

            print(f"Overall Statistics ({overall_date_range_str}): Avg={overall_avg_score:.2f}, Median={overall_median_score}, Mode(s)={overall_mode_score}, Total Scores={total_score_count}")

            # --- Prepare context for Final LLM Summary ---
            # Combine the LLM-generated batch summaries (which include trend analysis)
            # <<< CHANGE: Add date range context to batch summary strings fed to final prompt
            batch_summary_texts = [
                f"Batch {b['batch_num']} ({b.get('date_range', 'Unknown Range')}):\n{b['llm_summary']}"
                for b in batch_summaries_data if b.get('llm_summary') and not b['llm_summary'].startswith("[") # Exclude error messages
            ]
            combined_batch_summaries = "\n\n".join(batch_summary_texts)

            # --- Final Summary Prompt ---
            final_prompt = (
                f"You are creating a final sentiment analysis report about *nuclear fusion* based on news articles related to the United States, covering the period {overall_date_range_str}.\n"
                f"A total of {len(relevant_outputs)} relevant articles were analyzed, resulting in {total_score_count} valid sentiment scores.\n\n"
                f"--- Overall Calculated Statistics ({overall_date_range_str}) ---\n"
                f"Average Score (1-7 scale): {overall_avg_score:.2f}\n"
                f"Median Score: {overall_median_score}\n"
                f"Mode Score(s): {overall_mode_score}\n"
                f"Overall Score Frequency: {overall_freq_dist_str}\n\n"
                f"--- Summaries & Trend Observations from Batches ---\n"
                f"{combined_batch_summaries}\n\n"
                f"--- Final Task ---\n"
                f"Synthesize the batch summaries (including their trend observations) and the overall statistics into a cohesive final report (2-3 paragraphs).\n"
                f"1. Describe the predominant sentiment towards *nuclear fusion* (within US context) found in the articles.\n"
                f"2. Discuss any notable patterns or variations indicated by the statistics (e.g., distribution skew, comparison of mean/median/mode).\n"
                f"3. **Summarize Temporal Trends:** Based on the batch summaries, describe any overall trends in sentiment observed over the period {overall_date_range_str}. Was sentiment generally improving, declining, fluctuating, or stable? Mention specific periods if clear trends emerged in batches.\n"
                f"4. Provide a concluding statement on the overall sentiment landscape regarding fusion based on this data.\n"
                f"Focus on interpreting the provided information logically."
            )


            try:
                print("Generating final report...")
                final_response = chat(
                    model=MODEL,
                    options={
                        "num_ctx": LLM_CTX_FINAL,
                        "temperature": 0.6
                        },
                    messages=[{"role": "user", "content": final_prompt}]
                )
                final_report_text = final_response.get('message', {}).get('content', '').strip()
                if not final_report_text:
                     final_report_text = "[LLM returned empty final report]"
                     print("Warning: LLM returned empty final report")


            except Exception as e:
                 print(f"Error: LLM call failed for final summary: {e}")
                 final_report_text = f"[LLM Final Summary failed: {e}]"

            # Write final report
            out_file.write("\n--- FINAL SUMMARY ---\n\n")
            # <<< CHANGE: Include overall date range in final stats output
            out_file.write(f"Overall Statistics (Date Range: {overall_date_range_str}):\n")
            out_file.write(f"  Total Relevant Articles (US Context): {len(relevant_outputs)}\n")
            out_file.write(f"  Total Valid Fusion Sentiment Scores: {total_score_count}\n")
            out_file.write(f"  Average Score: {overall_avg_score:.2f}\n")
            out_file.write(f"  Median Score: {overall_median_score}\n")
            out_file.write(f"  Mode Score(s): {overall_mode_score}\n")
            out_file.write(f"  Score Distribution: {overall_freq_dist_str}\n\n")
            out_file.write(f"LLM Synthesized Report (including Temporal Trends):\n{final_report_text}\n")
            print("Final report generated.")

end_summary = time.time()
print(f"\n--- Pass 2 & 3 (Summarization) duration: {end_summary - start_summary:.2f} seconds ---")
print(f"--- Total execution time: {end_analysis - start_analysis:.2f} seconds ---")
print(f"Report saved to: {OUTPUT_REPORT}")
print(f"Intermediate data saved to: {OUTPUT_JSON}")