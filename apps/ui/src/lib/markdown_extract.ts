export function extractAssistantText(body: Record<string, unknown> | null | undefined): string {
  if (!body) return "";
  if (typeof body.content === "string") return body.content;
  if (typeof body.text === "string") return body.text;
  const message = body.message as { content?: unknown } | undefined;
  if (typeof message?.content === "string") return message.content;
  const choices = body.choices as Array<{ message?: { content?: unknown }; text?: string }> | undefined;
  const choiceContent = choices?.[0]?.message?.content;
  if (typeof choiceContent === "string") return choiceContent;
  if (Array.isArray(choiceContent)) {
    return choiceContent
      .map((item) =>
        typeof item === "string"
          ? item
          : (item as { text?: string; content?: string })?.text ||
            (item as { content?: string })?.content ||
            "",
      )
      .filter(Boolean)
      .join("\n");
  }
  if (typeof choices?.[0]?.text === "string") return choices[0].text;
  return "";
}
