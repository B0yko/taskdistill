Extract these fields from the customer's message:

- order_id (string): the order number exactly as written.
- order_date (string): the date the order was placed, as YYYY-MM-DD.
- total_amount (number or null): the order total as a plain number without a currency symbol, or null if the message does not state it.

Reply with one JSON object that has exactly these three keys, and nothing else: no Markdown, no explanation.
