# CAIS++ Website
by CAIS++

---
## About the Website
This website was built using `Jekyll` and `Liquid` and is hosted by `Github Pages`. Before editing, please read the respective documentations to gain a better understanding of how these frameworks work.

**Github Pages** [About Github Pages](http://docs.github.com/en/pages) <br>
**Jekyll** [Jekyll Documentation](https://jekyllrb.com/docs/) <br>
**Liquid** [Liquid Documentation](https://jekyllrb.com/docs/liquid/)

---
## Contributing
Please mind the current layout conventions and styles when adding to the website.
Please see the information on updating pages below. If you have any questions
about contribution, please message [Allen Chang](https://www.linkedin.com/in/cylumn/).

### 1. Projects
Each projects page will be under [projects/](/projects). The most recent page
should have the following redirect in their front matter:
```angular2html
redirect_from:
- /projects
```
Be sure to replace this from the current "most recent" project to ensure 
the correct redirects.

Project pages are generated from data are stored in [_data/projects.json](/_data/projects.json) and use the layout
[_layouts/project-page](/_layouts/project-page.html). You can view this layout to make any
updates necessary. 

[_data/people-project-metadata.json](/_data/people-project-metadata.json) contains metadata
to link projects from the "People" page to the "Project" page and should not be used
to generate project information.

#### Adding New Projects Checklist:
- [ ] Add project to [_data/projects.json](/_data/projects.json)
- [ ] Add contributors to `Official Roster`
- [ ] Add project to [_data/people-project-metadata.json](/_data/people-project-metadata.json)

### 2. People
The generated table in the people pages comes from uses 
[_includes/people-table.html](/_includes/people-table.html) and the data from 
[_data/roster.csv](/_data/roster.csv). **Do not** edit `roster.csv` directly.
This roster is generated from our `Official Roster` in our Google Drive. To generate
an updated `roster.csv`, please update the `Official Roster` and then follow the following
steps:
* Download the first sheet as a .csv file.
* Remove the first row (this row just contains additional information about each column)
* Replace `roster.csv` with the new .csv file.

The headers in the CSV file should be correct now. If any errors come from generating
the table, double check that the expected headers in [_includes/people-table.html](/_includes/people-table.html)
match the ones in the CSV.

### 3. Curriculum
The curriculum posts are located in [_posts/](/_posts) under a Liquid Blog post format.
Ordering, subsections, sources, and q&a of the curriculum lessons is 
contained within [_data/curriculum-metadata.json](/_data/curriculum-metadata.json).

### 4. Links
When adding links using `<a href="..."></a>`, please keep in mind whether we want
the user to **stay on or return to the same page** or be **navigated away from the current page**.
If we want the user to stay on or return to the same page, please make sure that
the link **opens in a new tab** using `target="_blank"`. Otherwise, opening the link
in the current tab is fine if we do not want to incentivize the user to stay on the
current page.

In general (with exceptions), external links should open on a new tab, 
whereas internal links should open on the same tab.