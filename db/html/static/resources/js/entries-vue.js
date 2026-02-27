/**
 * Entries data store (server-side pagination + filtering)
 * Fetches task entries from Django JSON API and renders the history table body.
 */

const entriesStore = {
    limit: 100,
    offset: 0,
    total: 0,
    hasMore: false,
    loading: false,
    templatesLoading: false,
    templatesLoaded: false,
    entries: [],
    templates: [],
    scrollThresholdPx: 250,
    scrollHandler: null,

    init() {
        this.attachInfiniteScroll();
        this.fetchEntries({ reset: true });
    },

    attachInfiniteScroll() {
        if (this.scrollHandler) return;
        this.scrollHandler = () => {
            if (!this.hasMore || this.loading) return;
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop || 0;
            const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
            const documentHeight = Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                document.body.offsetHeight,
                document.documentElement.offsetHeight
            );

            if (scrollTop + viewportHeight >= documentHeight - this.scrollThresholdPx) {
                this.fetchEntries({ reset: false });
            }
        };

        window.addEventListener('scroll', this.scrollHandler, { passive: true });
        window.addEventListener('resize', this.scrollHandler);
    },

    currentFilters() {
        const filters = (window.filtersApp && window.filtersApp.instance) ? window.filtersApp.instance : null;
        const selectedRig = filters ? filters.selectedRig : 'all';
        const useAllRigs = !filters || selectedRig === 'all';
        return {
            search: filters ? filters.searchQuery : '',
            task: filters && filters.selectedTask !== 'all' ? filters.selectedTask : '',
            subject: filters && filters.selectedSubject !== 'all' ? filters.selectedSubject : '',
            rig: (!useAllRigs && selectedRig !== 'all') ? selectedRig : '',
            end_date: filters ? (filters.endDate || '') : '',
            show_hidden: document.getElementById('toggle_visible')?.checked ? '1' : '0',
            show_all_rigs: useAllRigs ? '1' : '0',
        };
    },

    buildUrl({ reset }) {
        const params = new URLSearchParams(this.currentFilters());
        params.set('limit', String(this.limit));
        params.set('offset', String(reset ? 0 : this.offset));
        return `/exp_log/api/entries?${params.toString()}`;
    },

    setFilterOptions(options) {
        if (!window.filtersApp || !window.filtersApp.instance || !options) return;
        const app = window.filtersApp.instance;
        if (Array.isArray(options.tasks)) app.tasks = options.tasks;
        if (Array.isArray(options.subjects)) app.subjects = options.subjects;
        if (Array.isArray(options.rigs)) app.rigs = options.rigs;
    },

    updateFilterStats() {
        if (!window.filtersApp || !window.filtersApp.instance) return;
        window.filtersApp.instance.totalEntries = this.total;
        window.filtersApp.instance.visibleEntries = this.entries.length;
    },

    updateLoadMoreVisibility() {
        const loadMoreBtn = document.getElementById('entries_load_more');
        if (!loadMoreBtn) return;
        loadMoreBtn.style.display = 'none';
        loadMoreBtn.disabled = this.loading;
        loadMoreBtn.textContent = this.loading ? 'Loading…' : 'Load more';
    },

    ensureViewportFilled() {
        if (!this.hasMore || this.loading) return;
        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
        const documentHeight = Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight,
            document.body.offsetHeight,
            document.documentElement.offsetHeight
        );

        if (documentHeight <= viewportHeight + 40) {
            this.fetchEntries({ reset: false });
        }
    },

    async fetchTemplates() {
        if (this.templatesLoading) return;
        this.templatesLoading = true;
        try {
            const response = await fetch('/exp_log/api/templates', { credentials: 'same-origin' });
            if (!response.ok) throw new Error(`Templates API error: ${response.status}`);
            const data = await response.json();
            this.templates = data.templates || [];
            this.templatesLoaded = true;
        } catch (err) {
            console.error(err);
        } finally {
            this.templatesLoading = false;
        }
    },

    async handleTemplatesToggle(enabled) {
        if (enabled) {
            if (!this.templatesLoaded) {
                await this.fetchTemplates();
            }
        } else {
            this.templates = [];
            this.templatesLoaded = false;
        }
        this.renderRows();
    },

    async fetchEntries({ reset = true } = {}) {
        if (this.loading) return;
        this.loading = true;
        this.updateLoadMoreVisibility();
        try {
            const showTemplates = !!document.getElementById('toggle_templates')?.checked;
            if (showTemplates && (!this.templatesLoaded || reset)) {
                await this.fetchTemplates();
            }
            if (!showTemplates) {
                this.templates = [];
                this.templatesLoaded = false;
            }

            const response = await fetch(this.buildUrl({ reset }), { credentials: 'same-origin' });
            if (!response.ok) throw new Error(`Entries API error: ${response.status}`);
            const data = await response.json();

            if (reset) {
                this.entries = data.entries || [];
                this.offset = this.entries.length;
            } else {
                const newEntries = data.entries || [];
                this.entries = this.entries.concat(newEntries);
                this.offset += newEntries.length;
            }

            this.total = data.total || 0;
            this.hasMore = !!data.has_more;
            this.setFilterOptions(data.filter_options || {});
            this.renderRows();
            this.updateFilterStats();
            this.ensureViewportFilled();
        } catch (err) {
            console.error(err);
        } finally {
            this.loading = false;
            this.updateLoadMoreVisibility();
        }
    },

    renderRows() {
        const tbody = document.querySelector('#main tbody');
        if (!tbody) return;

        const newEntryRow = document.getElementById('newentry');
        const hiddenRow = tbody.querySelector('tr[name="hidden"]');

        const fragments = document.createDocumentFragment();

        if (document.getElementById('toggle_templates')?.checked) {
            this.templates.forEach((template, idx) => {
                const tr = document.createElement('tr');
                tr.id = `row${template.id}`;
                tr.className = 'template';
                tr.title = 'Template';
                if (!template.visible) tr.style.backgroundColor = 'gray';

                if (idx === 0) {
                    const tdHeader = document.createElement('td');
                    tdHeader.className = 'colDate';
                    tdHeader.setAttribute('rowspan', String(this.templates.length));
                    tdHeader.setAttribute('colspan', '3');
                    tdHeader.textContent = 'Templates';
                    tr.appendChild(tdHeader);
                }

                const tdName = document.createElement('td');
                tdName.className = 'colID';
                tdName.setAttribute('colspan', '2');
                tdName.style.textAlign = 'left';
                tdName.textContent = template.entry_name || '';

                const tdTask = document.createElement('td');
                tdTask.className = 'colTask';
                tdTask.setAttribute('colspan', '2');
                tdTask.style.textAlign = 'left';
                tdTask.textContent = template.task || '';

                tr.appendChild(tdName);
                tr.appendChild(tdTask);
                fragments.appendChild(tr);
            });
        }

        if (newEntryRow) fragments.appendChild(newEntryRow);
        if (hiddenRow) fragments.appendChild(hiddenRow);

        const dateCounts = this.entries.reduce((counts, entry) => {
            const key = entry.date || '';
            counts[key] = (counts[key] || 0) + 1;
            return counts;
        }, {});

        let previousDate = null;

        this.entries.forEach((entry) => {
            const tr = document.createElement('tr');
            tr.id = `row${entry.id}`;
            tr.title = entry.desc || '';
            if (entry.running) tr.className = 'running';
            if (!entry.visible) tr.style.backgroundColor = 'gray';

            const firstOfDay = (entry.date || '') !== previousDate;
            const dayClass = firstOfDay ? ' firstRowOfday' : '';

            if (firstOfDay) {
                const tdDate = document.createElement('td');
                tdDate.className = `colDate${dayClass}`;
                tdDate.setAttribute('rowspan', String(dateCounts[entry.date || ''] || 1));
                tdDate.textContent = entry.date || '';
                tr.appendChild(tdDate);
            }

            const tdTime = document.createElement('td');
            tdTime.className = `colTime${dayClass}`;
            tdTime.textContent = entry.time || '';

            const tdId = document.createElement('td');
            tdId.className = `colID${dayClass}`;
            tdId.textContent = entry.ui_id || '';

            const tdRig = document.createElement('td');
            tdRig.className = `colRig${dayClass}`;
            tdRig.textContent = entry.rig_name || '';

            const tdSubj = document.createElement('td');
            tdSubj.className = `colSubj${dayClass}`;
            tdSubj.textContent = entry.subject || '';

            const tdTask = document.createElement('td');
            tdTask.className = `colTask${dayClass}`;
            tdTask.textContent = entry.task || '';

            const tdDesc = document.createElement('td');
            tdDesc.className = `colShortDesc${dayClass}`;
            tdDesc.textContent = entry.desc || '';

            tr.appendChild(tdTime);
            tr.appendChild(tdId);
            tr.appendChild(tdRig);
            tr.appendChild(tdSubj);
            tr.appendChild(tdTask);
            tr.appendChild(tdDesc);

            previousDate = entry.date || '';

            fragments.appendChild(tr);
        });

        while (tbody.firstChild) {
            tbody.removeChild(tbody.firstChild);
        }

        tbody.appendChild(fragments);
    },
};

window.entriesStore = entriesStore;
