import { describe, expect, it } from "vitest";
import { renderMarkdown } from "./markdown";

const SAMPLE = `Here is a message containing all standard Markdown notations.

---

# Heading Level 1
## Heading Level 2
### Heading Level 3
#### Heading Level 4
##### Heading Level 5
###### Heading Level 6
This is a paragraph with **bold text**, *italic text*, and ***bold and italic*** text. You can also use __bold__ and _italic_. Here is some \`inline code\` and ~~strikethrough~~ text. This paragraph demonstrates a manual  
line break.

> This is a blockquote.
>
> > This is a nested blockquote.

Unordered List:

- Item 1
- Item 2
  - Sub-item A
  - Sub-item B
* Another style
+ Yet another style

Ordered List:

1. First item
2. Second item
   1. Nested item one
   2. Nested item two
3. Third item
Horizontal Rule:
---

This is a [link to OpenAI](https://openai.com)
[This is a link with a title](https://www.example.com "Example Website")

<https://www.google.com>
<fake@example.com>

Here is an image:
![Placeholder Image](https://via.placeholder.com/150 "Image Title")

Here is a footnote reference[^1].

[^1]: This is the footnote content.

| Left-aligned | Center-aligned | Right-aligned |
| :--- | :---: | ---: |
| Cell 1 | Cell 2 | Cell 3 |
| Cell 4 | Cell 5 | Cell 6 |
[x] Completed task
[ ] Incomplete task
[ ] Another task
Here is a definition list:
Term 1
: Definition 1

Term 2
: Definition 2a
: Definition 2b

\`\`\`json
{
  "firstName": "John",
  "lastName": "Doe",
  "age": 25
}
\`\`\`
\`\`\`javascript
function greet(name) {
  console.log("Hello, " + name + "!");
}
greet("World");
\`\`\`
The \`printf()\` function outputs text.
Escape an asterisk: \\*not italic\\*.

Here is some regular text.`;

describe("renderMarkdown", () => {
  it("renders standard markdown constructs from the parity sample", () => {
    const html = renderMarkdown(SAMPLE);
    expect(html).toContain("<h1>Heading Level 1</h1>");
    expect(html).toContain("<h6>Heading Level 6</h6>");
    expect(html).toContain("<strong>bold text</strong>");
    expect(html).toContain("<em>italic text</em>");
    expect(html).toContain("<strong><em>bold and italic</em></strong>");
    expect(html).toContain("<del>strikethrough</del>");
    expect(html).toContain("<code>inline code</code>");
    expect(html).toContain("<br>");
    expect(html).toContain("<blockquote>");
    expect(html).toContain("This is a nested blockquote");
    expect(html).toContain("<ul>");
    expect(html).toContain("Sub-item A");
    expect(html).toContain("<ol>");
    expect(html).toContain("Nested item one");
    expect(html).toContain("<hr>");
    expect(html).toContain('href="https://openai.com"');
    expect(html).toContain('title="Example Website"');
    expect(html).toContain('href="https://www.google.com"');
    expect(html).toContain('href="mailto:fake@example.com"');
    expect(html).toContain('<img src="https://via.placeholder.com/150"');
    expect(html).toContain('class="footnote-ref"');
    expect(html).toContain('id="fn-1"');
    expect(html).toContain("This is the footnote content");
    expect(html).toContain('<table class="md-table">');
    expect(html).toContain("text-align:center");
    expect(html).toContain("text-align:right");
    expect(html).toContain("task-item is-checked");
    expect(html).toContain("task-item");
    expect(html).toContain("<dl>");
    expect(html).toContain("<dt>Term 1</dt>");
    expect(html).toContain("Definition 2b");
    expect(html).toContain('class="language-json"');
    expect(html).toContain("&quot;firstName&quot;: &quot;John&quot;");
    expect(html).toContain('class="language-javascript"');
    expect(html).toContain('greet(&quot;World&quot;)');
    expect(html).toContain("<code>printf()</code>");
    expect(html).toContain("*not italic*");
    expect(html).not.toContain("\\*not italic\\*");
    expect(html).not.toContain("<em>not italic</em>");
  });

  it("renders safe inline HTML and footnote references", () => {
    const html = renderMarkdown(
      `Here is a <span style="color:red;">HTML span tag</span> for raw HTML, and a footnote reference[^1].

[^1]: This is the footnote content.`,
    );
    expect(html).toContain('<span style="color: red">HTML span tag</span>');
    expect(html).not.toContain("&lt;span");
    expect(html).toContain('class="footnote-ref"');
    expect(html).toContain("This is the footnote content");
  });
});
