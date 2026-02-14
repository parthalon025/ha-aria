import { useState, useEffect, useRef } from 'preact/hooks';
import { fetchJson, putJson, postJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Group configs by category for collapsible sections. */
function groupByCategory(configs) {
  const groups = {};
  for (const c of configs) {
    const cat = c.category || 'Other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(c);
  }
  return groups;
}

function ParamControl({ config, onUpdate }) {
  const [value, setValue] = useState(config.value);
  const [saving, setSaving] = useState(false);
  const timerRef = useRef(null);
  const isDefault = config.value === config.default_value;

  useEffect(() => { setValue(config.value); }, [config.value]);

  function debouncedSave(newVal) {
    setValue(newVal);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setSaving(true);
      try {
        await putJson(`/api/config/${config.key}`, { value: String(newVal) });
        onUpdate();
      } catch (e) {
        console.error('Config save failed:', e);
      } finally {
        setSaving(false);
      }
    }, 500);
  }

  async function handleReset() {
    setSaving(true);
    try {
      await postJson(`/api/config/reset/${config.key}`, {});
      setValue(config.default_value);
      onUpdate();
    } catch (e) {
      console.error('Config reset failed:', e);
    } finally {
      setSaving(false);
    }
  }

  const vtype = config.value_type;

  return (
    <div class="flex flex-col sm:flex-row sm:items-center gap-2 py-3" style="border-bottom: 1px solid var(--border-subtle)">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="text-sm font-medium" style="color: var(--text-secondary)">{config.label || config.key}</span>
          {saving && <span class="text-xs" style="color: var(--accent)">Saving...</span>}
        </div>
        {config.description && (
          <p class="text-xs mt-0.5" style="color: var(--text-tertiary)">{config.description}</p>
        )}
      </div>
      <div class="flex items-center gap-2 flex-shrink-0">
        {vtype === 'number' && (
          <div class="flex items-center gap-2">
            <input
              type="range"
              min={config.min_value ?? 0}
              max={config.max_value ?? 100}
              step={config.step ?? 1}
              value={value}
              onInput={(e) => debouncedSave(e.target.value)}
              class="w-28"
              style="accent-color: var(--accent)"
            />
            <span class="text-sm w-16 text-right data-mono" style="color: var(--text-secondary)">{value}</span>
          </div>
        )}
        {vtype === 'boolean' && (
          <button
            class="relative w-10 h-5 rounded-full transition-colors"
            style={value === 'true' || value === '1' ? 'background: var(--accent)' : 'background: var(--bg-inset)'}
            onClick={() => debouncedSave(value === 'true' || value === '1' ? 'false' : 'true')}
          >
            <span class={`absolute top-0.5 w-4 h-4 rounded-full shadow transition-transform ${
              value === 'true' || value === '1' ? 'translate-x-5' : 'translate-x-0.5'
            }`} style="background: var(--bg-surface)" />
          </button>
        )}
        {vtype === 'string' && (
          <input
            type="text"
            value={value}
            onInput={(e) => debouncedSave(e.target.value)}
            class="t-input px-2 py-1 text-sm w-48"
          />
        )}
        {vtype === 'select' && (
          <select
            value={value}
            onChange={(e) => debouncedSave(e.target.value)}
            class="t-input px-2 py-1 text-sm"
          >
            {(config.options || '').split(',').map((o) => (
              <option key={o.trim()} value={o.trim()}>{o.trim()}</option>
            ))}
          </select>
        )}
        <button
          onClick={handleReset}
          disabled={isDefault}
          class="text-xs px-2 py-1"
          style={`border-radius: var(--radius); ${
            isDefault
              ? 'color: var(--text-tertiary); cursor: default;'
              : 'color: var(--accent); cursor: pointer;'
          }`}
        >
          Reset
        </button>
      </div>
    </div>
  );
}

function CategorySection({ category, configs, onUpdate }) {
  const [open, setOpen] = useState(true);

  return (
    <section class="t-frame" data-label={category.toLowerCase()}>
      <button
        class="w-full flex items-center justify-between px-4 py-3 text-left"
        onClick={() => setOpen(!open)}
      >
        <h3 class="text-sm font-bold" style="color: var(--text-secondary)">{category}</h3>
        <span class="text-xs" style="color: var(--text-tertiary)">{open ? '\u25B2' : '\u25BC'} {configs.length} param{configs.length !== 1 ? 's' : ''}</span>
      </button>
      {open && (
        <div class="px-4 pb-3">
          {configs.map((c) => (
            <ParamControl key={c.key} config={c} onUpdate={onUpdate} />
          ))}
        </div>
      )}
    </section>
  );
}

export default function Settings() {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function fetchConfigs() {
    try {
      const data = await fetchJson('/api/config');
      setConfigs(data.configs || []);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchConfigs(); }, []);

  if (loading && configs.length === 0) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Settings</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Configure shadow engine, activity monitor, and data quality parameters.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Settings</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Configure shadow engine, activity monitor, and data quality parameters.</p>
        </div>
        <ErrorState error={error} onRetry={fetchConfigs} />
      </div>
    );
  }

  const groups = groupByCategory(configs);
  const modified = configs.filter((c) => c.value !== c.default_value).length;

  return (
    <div class="space-y-6 animate-page-enter">
      <div class="t-section-header" style="padding-bottom: 8px;">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Settings</h1>
        <p class="text-sm" style="color: var(--text-tertiary)">
          {configs.length} parameters across {Object.keys(groups).length} categories.
          {modified > 0 && <span style="color: var(--accent)"> {modified} modified from defaults.</span>}
        </p>
      </div>

      <div class="space-y-4">
        {Object.entries(groups).map(([cat, items]) => (
          <CategorySection key={cat} category={cat} configs={items} onUpdate={fetchConfigs} />
        ))}
      </div>
    </div>
  );
}
