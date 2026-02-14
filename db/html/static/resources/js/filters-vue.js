/**
 * Vue.js based Filters Component for Experiment List
 * Provides real-time search and filtering for the task entry table
 * 
 * Replaces static filters with reactive, responsive filtering
 */

const filtersApp = {
    el: '#filters_vue',
    data() {
        return {
            searchQuery: '',
            selectedTask: 'all',
            selectedSubject: 'all',
            selectedRig: 'all',
            showTemplates: false,
            showHidden: false,
            showAllRigs: false,
            
            // Available filter options (will be populated from page data)
            tasks: [],
            subjects: [],
            rigs: [],
            
            // Statistics
            totalEntries: 0,
            visibleEntries: 0,
        };
    },
    
    computed: {
        /**
         * Get all visible table rows based on current filters
         */
        activeFilters() {
            return {
                search: this.searchQuery.toLowerCase(),
                task: this.selectedTask !== 'all' ? this.selectedTask : null,
                subject: this.selectedSubject !== 'all' ? this.selectedSubject : null,
                rig: this.selectedRig !== 'all' ? this.selectedRig : null,
                templates: this.showTemplates,
                hidden: this.showHidden,
                allRigs: this.showAllRigs,
            };
        },
        
        /**
         * Human-readable filter summary
         */
        filterSummary() {
            const filters = [];
            if (this.searchQuery) filters.push(`Search: "${this.searchQuery}"`);
            if (this.selectedTask !== 'all') filters.push(`Task: ${this.selectedTask}`);
            if (this.selectedSubject !== 'all') filters.push(`Subject: ${this.selectedSubject}`);
            if (this.selectedRig !== 'all') filters.push(`Rig: ${this.selectedRig}`);
            
            if (filters.length === 0) return 'All filters active';
            return filters.join(' • ');
        },
        
        /**
         * Check if any filters are active
         */
        hasActiveFilters() {
            return this.searchQuery || 
                   this.selectedTask !== 'all' || 
                   this.selectedSubject !== 'all' || 
                   this.selectedRig !== 'all';
        }
    },
    
    methods: {
        /**
         * Initialize filter options from available data
         */
        initializeFilters() {
            // Extract unique values from table rows
            const taskSet = new Set();
            const subjectSet = new Set();
            const rigSet = new Set();
            
            // Get all data rows (exclude header, new entry row, and templates)
            const rows = document.querySelectorAll('#main tbody tr');
            
            rows.forEach(row => {
                // Skip special rows
                if (row.id === 'newentry' || row.id === 'te_table_header') return;
                if (row.classList.contains('template')) return;
                if (row.name === 'hidden') return;
                
                const cells = row.querySelectorAll('td');
                if (cells.length >= 6) {
                    // Find the actual data cells (might be in different positions due to rowspan)
                    // Look for colTask, colSubj, colRig classes
                    const rigCell = row.querySelector('.colRig') || cells[2];
                    const subjCell = row.querySelector('.colSubj') || cells[3];
                    const taskCell = row.querySelector('.colTask') || cells[4];
                    
                    const rigText = rigCell?.textContent?.trim();
                    const subjText = subjCell?.textContent?.trim();
                    const taskText = taskCell?.textContent?.trim();
                    
                    if (rigText && rigText !== 'Hidden' && rigText !== 'All rigs') rigSet.add(rigText);
                    if (subjText && subjText !== 'Hidden') subjectSet.add(subjText);
                    if (taskText && taskText !== 'Hidden') taskSet.add(taskText);
                }
            });
            
            this.tasks = Array.from(taskSet).sort();
            this.subjects = Array.from(subjectSet).sort();
            this.rigs = Array.from(rigSet).sort();
            
            // Count non-template rows
            const countedRows = document.querySelectorAll('#main tbody tr:not(.template):not([name="hidden"])');
            let dataRowCount = 0;
            countedRows.forEach(row => {
                if (row.id && !['te_table_header', 'newentry'].includes(row.id)) {
                    dataRowCount++;
                }
            });
            this.totalEntries = dataRowCount;
        },
        
        /**
         * Check if a row matches current filters
         */
        rowMatchesFilters(row) {
            // Always show header and new entry row
            if (row.id === 'te_table_header' || row.id === 'newentry') return true;
            if (row.name === 'hidden') return true;
            
            // Skip templates if not showing
            if (!this.showTemplates && row.classList.contains('template')) return false;
            
            // Get row data using column classes for better accuracy
            const rigCell = row.querySelector('.colRig');
            const subjCell = row.querySelector('.colSubj');
            const taskCell = row.querySelector('.colTask');
            const descCell = row.querySelector('.colShortDesc');
            const dateCell = row.querySelector('.colDate');
            const timeCell = row.querySelector('.colTime');
            
            // Fallback to indexed cells if class-based selectors fail
            const cells = row.querySelectorAll('td');
            const rowData = {
                rig: rigCell?.textContent?.trim() || cells[2]?.textContent?.trim() || '',
                subject: subjCell?.textContent?.trim() || cells[3]?.textContent?.trim() || '',
                task: taskCell?.textContent?.trim() || cells[4]?.textContent?.trim() || '',
                desc: descCell?.textContent?.trim() || cells[6]?.textContent?.trim() || '',
                date: dateCell?.textContent?.trim() || cells[0]?.textContent?.trim() || '',
                time: timeCell?.textContent?.trim() || cells[1]?.textContent?.trim() || '',
            };
            
            // Check if hidden (gray background)
            const isHidden = row.style.backgroundColor === 'gray';
            if (isHidden && !this.showHidden) return false;
            
            // Check filters
            if (this.selectedTask !== 'all' && rowData.task !== this.selectedTask) {
                return false;
            }
            
            if (this.selectedSubject !== 'all' && rowData.subject !== this.selectedSubject) {
                return false;
            }
            
            if (this.selectedRig !== 'all' && rowData.rig !== this.selectedRig) {
                return false;
            }
            
            // Check search query
            if (this.searchQuery) {
                const query = this.searchQuery.toLowerCase();
                const searchableText = [
                    rowData.task,
                    rowData.subject,
                    rowData.rig,
                    rowData.desc,
                    rowData.date,
                    rowData.time
                ].join(' ').toLowerCase();
                
                if (!searchableText.includes(query)) {
                    return false;
                }
            }
            
            return true;
        },
        
        /**
         * Apply filters to table rows
         */
        applyFilters() {
            const rows = document.querySelectorAll('#main tbody tr');
            let visibleCount = 0;
            
            rows.forEach(row => {
                // Always show special rows
                if (row.id === 'te_table_header' || row.id === 'newentry' || row.name === 'hidden') {
                    row.style.display = '';
                    return;
                }
                
                const shouldShow = this.rowMatchesFilters(row);
                row.style.display = shouldShow ? '' : 'none';
                
                // Count visible data rows only
                if (shouldShow && row.id && !['te_table_header', 'newentry'].includes(row.id)) {
                    visibleCount++;
                }
            });
            
            this.visibleEntries = visibleCount;
        },
        
        /**
         * Clear all filters
         */
        clearFilters() {
            this.searchQuery = '';
            this.selectedTask = 'all';
            this.selectedSubject = 'all';
            this.selectedRig = 'all';
            this.showTemplates = false;
            this.showHidden = false;
            this.applyFilters();
        },
        
        /**
         * Handle search input
         */
        onSearchInput() {
            this.applyFilters();
        },
        
        /**
         * Handle filter selection
         */
        onFilterChange() {
            this.applyFilters();
        },
        
        /**
         * Toggle hidden entries visibility
         */
        toggleHidden() {
            this.showHidden = !this.showHidden;
            this.applyFilters();
        },
        
        /**
         * Toggle templates visibility
         */
        toggleTemplates() {
            this.showTemplates = !this.showTemplates;
            this.applyFilters();
        },
        
        /**
         * Toggle all rigs visibility
         */
        toggleAllRigs() {
            this.showAllRigs = !this.showAllRigs;
            // Reset rig filter if not showing all rigs
            if (!this.showAllRigs) {
                this.selectedRig = 'all';
            }
            this.applyFilters();
        }
    },
    
    mounted() {
        console.log("Filters Vue app mounted");
        this.initializeFilters();
        this.applyFilters();
        
        // Watch for table updates (when new rows are added via AJAX)
        const observer = new MutationObserver(() => {
            this.initializeFilters();
            this.applyFilters();
        });
        
        const table = document.querySelector('#main tbody');
        if (table) {
            observer.observe(table, {
                childList: true,
                attributes: true,
                subtree: true
            });
        }
    },
    
    watch: {
        // Watch for external changes to the checkbox states
        showHidden(newVal) {
            const checkbox = document.getElementById('toggle_visible');
            if (checkbox) checkbox.checked = newVal;
        },
        showTemplates(newVal) {
            const checkbox = document.getElementById('toggle_templates');
            if (checkbox) checkbox.checked = newVal;
        },
        showAllRigs(newVal) {
            const checkbox = document.getElementById('toggle_rigs');
            if (checkbox) checkbox.checked = newVal;
        }
    }
};

// Export for use
window.filtersApp = filtersApp;

/**
 * Bridge function to sync external checkbox changes
 */
window.syncExternalFilters = function() {
    if (filtersApp.instance) {
        const visibleCheckbox = document.getElementById('toggle_visible');
        const templatesCheckbox = document.getElementById('toggle_templates');
        const rigsCheckbox = document.getElementById('toggle_rigs');
        
        if (visibleCheckbox) {
            filtersApp.instance.showHidden = visibleCheckbox.checked;
        }
        if (templatesCheckbox) {
            filtersApp.instance.showTemplates = templatesCheckbox.checked;
        }
        if (rigsCheckbox) {
            filtersApp.instance.showAllRigs = rigsCheckbox.checked;
        }
    }
};
