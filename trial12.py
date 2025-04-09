    def generate_result_summary(results):
        """Generate a meaningful summary of the query results tailored to specific queries like min/max or highest kWh savings."""
        try:
            # Convert DataFrame to a readable string format for summarization
            results_text = results.to_string(index=False)
            initial_summary = summarize(results_text)
            if not initial_summary:
                return "⚠️ Unable to generate an initial summary."

            # Check if the query is about kWh savings (highest, min, or max), only if current_query exists
            if (st.session_state.current_query is not None and 
                "kwh savings" in st.session_state.current_query.lower()):
                # Check for correct column names (case-insensitive)
                if 'county' in results.columns.str.lower() and 'kwh_savings' in results.columns.str.lower():
                    # Map to actual column names
                    county_col = [col for col in results.columns if col.lower() == 'county'][0]
                    kwh_savings_col = [col for col in results.columns if col.lower() == 'kwh_savings'][0]

                    # Handle min and max query
                    if "min" in st.session_state.current_query.lower() and "max" in st.session_state.current_query.lower():
                        max_row = results.loc[results[kwh_savings_col].idxmax()]
                        min_row = results.loc[results[kwh_savings_col].idxmin()]
                        max_county = max_row[county_col]
                        max_value = max_row[kwh_savings_col]
                        min_county = min_row[county_col]
                        min_value = min_row[kwh_savings_col]
                        return (f"The county with the highest kWh savings is {max_county} with approximately {max_value:,.0f} kWh, "
                                f"and the county with the lowest kWh savings is {min_county} with approximately {min_value:,.0f} kWh.")
                    # Handle highest only
                    elif "highest" in st.session_state.current_query.lower():
                        max_row = results.loc[results[kwh_savings_col].idxmax()]
                        county = max_row[county_col]
                        kwh_value = max_row[kwh_savings_col]
                        return f"The highest kilowatt-hours (kWh) savings county is {county} and the value is approximately {kwh_value:,.0f}."
                else:
                    # Fallback to summarize and complete for natural language conversion
                    prompt = f"Convert the following query results into a natural language summary:\n\n{initial_summary}"
                    natural_language_summary = complete(prompt)
                    if natural_language_summary:
                        return natural_language_summary
                    else:
                        return "⚠️ Unable to generate a natural language summary from results."

            # Fallback to generic summary for other queries
            prompt = f"Provide a concise, meaningful summary of the following query results:\n\n{initial_summary}"
            meaningful_summary = complete(prompt)
            if meaningful_summary:
                return meaningful_summary
            else:
                return "⚠️ Unable to generate a meaningful summary."
        except Exception as e:
            return f"⚠️ Summary generation failed: {str(e)}"
