# Website

The site is static HTML and CSS in `public/`. It has no build step or JavaScript
dependencies. Serve that directory at the domain root.

To preview it locally, run this from the repository root:

```sh
python3 -m http.server 8000 --directory website/public
```

Open http://localhost:8000. The blog is at `/blog/`.

## Adding a post

Use plain ASCII text, including straight quotes and apostrophes.

1. Create `public/blog/<slug>/index.html`, using the existing article's head,
   navigation and footer. Use `public/style.css` for shared styling.
2. Set the title, description, canonical URL and date. Give sections stable IDs.
3. Add the post to `public/blog/index.html`, newest first, and update the latest
   article under "From the blog" in `public/index.html`.
4. Keep any downloadable examples beside the article. Verify they match the
   displayed code and run them before publishing.

Check the homepage, blog index and article at desktop and mobile widths. Code
blocks scroll horizontally; the rest of the page should fit the viewport.
