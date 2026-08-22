import { useState, useRef, useEffect } from 'react';
import { ChevronDown, X } from 'lucide-react';

interface MultiSelectDropdownProps {
  label: string;
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  onSelectAll: () => void;
  onRemove: (value: string) => void;
  columns?: number;
  placeholder?: string;
}

export function MultiSelectDropdown({
  label, options, selected, onToggle, onSelectAll, onRemove, columns = 1,
  placeholder = 'No items selected',
}: MultiSelectDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const allSelected = options.length > 0 && selected.length === options.length;
  const colsCls = columns === 2 ? 'grid grid-cols-2 gap-1 p-2'
                : columns === 3 ? 'grid grid-cols-3 gap-1 p-2'
                : columns === 4 ? 'grid grid-cols-4 gap-1 p-2'
                : '';

  return (
    <div className="flex items-start gap-4">
      <label className="text-foreground-muted text-xs uppercase tracking-wider font-medium w-32 text-right flex-shrink-0 pt-2.5">
        {label}
      </label>
      <div className="flex-1 min-w-0 relative" ref={dropdownRef}>
        {/* Selected chips */}
        <div className="flex flex-wrap gap-1.5 mb-2 min-h-[28px]">
          {selected.length > 0 ? (
            selected.map((item) => (
              <span
                key={item}
                className="inline-flex items-center gap-1.5 bg-primary-muted border border-primary/30 text-primary px-2 py-0.5 rounded-md text-xs"
              >
                <span className="font-mono">{item}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); onRemove(item); }}
                  className="hover:text-foreground transition-colors"
                  aria-label={`Remove ${item}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))
          ) : (
            <div className="text-foreground-faint text-xs py-1">{placeholder}</div>
          )}
        </div>

        {/* Trigger */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors hover:border-border-strong flex items-center justify-between"
        >
          <span className="text-foreground-muted">
            {selected.length === 0 ? placeholder.replace('No ', 'Select ').replace(' selected', '…')
                                   : `${selected.length} selected`}
          </span>
          <ChevronDown className={`w-4 h-4 text-foreground-subtle transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>

        {/* Menu */}
        {isOpen && (
          <div className="absolute z-30 mt-1 w-full bg-surface-2 border border-border rounded-md shadow-2xl shadow-black/60 max-h-72 overflow-y-auto">
            <div
              onClick={onSelectAll}
              className="px-3 py-2.5 hover:bg-surface-3 cursor-pointer border-b border-border-subtle flex items-center gap-3 sticky top-0 bg-surface-2"
            >
              <input
                type="checkbox"
                checked={allSelected}
                onChange={() => {}}
                className="w-3.5 h-3.5 accent-primary"
              />
              <span className="text-foreground text-sm font-medium">
                {allSelected ? 'Deselect all' : 'Select all'}
              </span>
            </div>

            <div className={colsCls}>
              {options.map((option) => (
                <div
                  key={option}
                  onClick={() => onToggle(option)}
                  className={`cursor-pointer flex items-center gap-2 ${
                    columns > 1 ? 'px-2 py-1.5 hover:bg-surface-3 rounded' : 'px-3 py-2 hover:bg-surface-3'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(option)}
                    onChange={() => {}}
                    className="w-3.5 h-3.5 accent-primary flex-shrink-0"
                  />
                  <span className="text-foreground text-sm whitespace-nowrap overflow-hidden text-ellipsis">{option}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
