Automated Item Generation Using GPT-3
================

## GitHub Documents

This is an R Markdown format used for publishing markdown documents to
GitHub. When you click the **Knit** button all R code chunks are run and
a markdown file (.md) suitable for publishing to GitHub is generated.

## Including Code

You can include R code in the document as follows:

``` python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="davinci",
  prompt="Objective: This program generates items that measure Big Five personality traits.\n\ntrait: neuroticism\nnumber of items to generate: 8\n1. I am often down in the dumps .\n2. I have frequent mood swings .\n3. I dislike myself .\n4. I often feel blue .\n5. I panic easily .\n6. I rarely get irritated .\n7. I am not easily bothered by things .\n8. I seldom feel blue .\n###\ntrait: extraversion\nnumber of items to generate: 10\n1. I make friends easily .\n2. I feel comfortable around people .\n3. I am skilled in handling social situations .\n4. I am the life of the party .\n5. I know how to captivate people .\n6. I keep in the background .\n7. I have little to say .\n8. I would describe my experiences as somewhat dull .\n9. I do not like to draw attention to myself .\n10. I do not talk a lot .\n###\ntrait: openness\nnumber of items to generate: 2\n1. I believe in the importance of art .\n2. I have a vivid i imagination .\n###\ntrait: agreeableness\nnumber of items to generate: 15\n1. I am considerate and kind to almost everyone .\n2. I get along well with others .\n3. I rarely start quarrels with others .\n4. I am helpful and unselfish with others .\n5. I get along well with most people .\n6. People usually like me .\n7. I am not particularly quarrelsome .\n8. I do not find it difficult to get along with others .\n9. People often accuse me of being too nice .\n10. I have a forgiving nature .\n11. It does not bother me if others do not respect me .\n12. I like most people at first sight .\n13. If people are nice to me, I am nice in return .\n14. It does not take a lot to make me happy .\n15. It is important to be kind to everyone you meet . ",
  temperature=0.6,
  max_tokens=200,
  top_p=0.8,
  frequency_penalty=0.5,
  presence_penalty=0.5,
  stop=["###"]
)
```

## Including Plots

You can also embed plots, for example:

![](automated-item-generation-with-transformers_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
