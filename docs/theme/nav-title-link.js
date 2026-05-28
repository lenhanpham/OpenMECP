/// Makes the book title in the top menu bar a clickable link to the home page.
/// mdBook renders the title as a plain <h1 class="menu-title"> in the nav bar;
/// this script wraps it in an <a> tag using the page-relative `path_to_root`
/// variable that mdBook injects into every page.
(function () {
  'use strict';

  function linkifyMenuTitle() {
    var h1 = document.querySelector('.menu-bar .menu-title');
    if (!h1 || h1.querySelector('a')) { return; } // not found or already a link

    // mdBook injects `var path_to_root = "..."` in each page's <script> block
    var root = (typeof window.path_to_root !== 'undefined') ? window.path_to_root : './';

    var a = document.createElement('a');
    a.href = root + 'index.html';
    a.className = 'menu-title-home';
    a.textContent = h1.textContent.trim();
    a.title = 'Home';

    // Inject minimal hover style so the link feels interactive
    var style = document.createElement('style');
    style.textContent = [
      'a.menu-title-home { color: inherit; text-decoration: none; }',
      'a.menu-title-home:hover { opacity: 0.75; }'
    ].join('\n');
    document.head.appendChild(style);

    h1.textContent = '';
    h1.appendChild(a);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', linkifyMenuTitle);
  } else {
    linkifyMenuTitle();
  }
}());
