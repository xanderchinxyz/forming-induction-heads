from IPython.display import HTML, display
import json
import torch
import uuid

def interactive_attention_viewer(
    models,
    sequence: torch.Tensor,
    seq_len: int,
    model_names: list = None,
    max_tokens: int = 40,
):
    """
    Create an interactive attention head viewer (TransformerLens/circuitsvis style).

    Features:
    - Attention pattern heatmap on the left
    - Head selector thumbnails (hover to focus, click to lock)
    - Token row with gradient highlighting based on attention
    - Model selector dropdown (when multiple models provided)
    - Induction accuracy (accuracy after first sequence repeat)

    Args:
        models: A single transformer model or a list of transformer models
        sequence: Input sequence tensor of shape [1, total_len]
        seq_len: Length of the repeating sequence (for induction accuracy calculation)
        model_names: List of names for display (optional, defaults to "Model 1", "Model 2", etc.)
        max_tokens: Maximum number of tokens to display
    """
    # Normalize models to a list
    if not isinstance(models, (list, tuple)):
        models = [models]

    # Generate default model names if not provided
    if model_names is None:
        if len(models) == 1:
            model_names = ["Model"]
        else:
            model_names = [f"Model {i+1}" for i in range(len(models))]

    show_model_dropdown = len(models) > 1

    # Collect attention data, predictions, and accuracies for all models
    all_models_attention_data = {}
    all_models_head_lists = {}
    all_models_predictions = {}  # top-5 predictions per position
    all_models_accuracies = {}   # induction accuracy per model

    for model_idx, model in enumerate(models):
        model_key = model_names[model_idx]

        # Run forward pass to populate attention weights and get logits
        model.eval()
        with torch.no_grad():
            input_seq = sequence[:, :-1]
            logits = model(input_seq)  # [1, seq_len-1, vocab_size]

            # Compute top-5 predictions for each position
            probs = torch.softmax(logits[0], dim=-1)  # [seq_len-1, vocab_size]
            top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)  # [seq_len-1, 5]

            # Store predictions: list of {tokens: [...], probs: [...]} per position
            predictions = []
            for pos in range(min(max_tokens - 1, top5_indices.shape[0])):
                pred_tokens = top5_indices[pos].cpu().tolist()
                pred_probs = top5_probs[pos].cpu().tolist()
                predictions.append({
                    "tokens": pred_tokens,
                    "probs": [round(p, 3) for p in pred_probs]
                })

            # Compute induction accuracy: only measure after first sequence repeat
            # Positions 0 to seq_len-1 are first occurrence, seq_len onwards is where induction matters
            target_tokens = sequence[0, 1:max_tokens].cpu()  # actual next tokens
            predicted_tokens = torch.argmax(logits[0, :max_tokens-1], dim=-1).cpu()

            # Only count positions >= seq_len (after first repeat)
            induction_start = seq_len
            if induction_start < len(predicted_tokens):
                induction_predicted = predicted_tokens[induction_start:]
                induction_targets = target_tokens[induction_start:len(predicted_tokens)]
                correct = (induction_predicted == induction_targets).float()
                accuracy = correct.mean().item()
            else:
                accuracy = 0.0

        n_layers = model.n_layers
        n_heads = model.attention_blocks[0].n_heads

        # Collect attention patterns for all heads
        attention_data = {}
        head_list = []
        for layer_idx in range(n_layers):
            attn_weights = model.attention_blocks[layer_idx].attn_weights[0]  # [n_heads, seq, seq]
            for head_idx in range(n_heads):
                head_name = f"Head {head_idx}" if n_layers == 1 else f"L{layer_idx}H{head_idx}"
                pattern = attn_weights[head_idx, :max_tokens, :max_tokens].cpu().numpy().tolist()
                attention_data[head_name] = pattern
                head_list.append(head_name)

        all_models_attention_data[model_key] = attention_data
        all_models_head_lists[model_key] = head_list
        all_models_predictions[model_key] = predictions
        all_models_accuracies[model_key] = round(accuracy * 100, 1)

    tokens = sequence[0, :max_tokens].cpu().tolist()
    tokens_json = json.dumps(tokens)
    all_attention_json = json.dumps(all_models_attention_data)
    all_head_lists_json = json.dumps(all_models_head_lists)
    all_predictions_json = json.dumps(all_models_predictions)
    all_accuracies_json = json.dumps(all_models_accuracies)
    model_names_json = json.dumps(model_names)
    show_model_dropdown_json = json.dumps(show_model_dropdown)
    viewer_id = uuid.uuid4().hex[:12]
    
    html = f"""
    <style>
        .cv-container-{viewer_id} {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #2d2d2d;
            padding: 16px;
            border-radius: 4px;
            color: #fff;
        }}
        
        .cv-main-{viewer_id} {{
            display: flex;
            gap: 24px;
            align-items: flex-start;
        }}
        
        .cv-left-{viewer_id}, .cv-right-{viewer_id} {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .cv-section-title-{viewer_id} {{
            font-size: 13px;
            font-weight: 600;
            color: #fff;
        }}
        
        .cv-section-subtitle-{viewer_id} {{
            font-size: 11px;
            color: #999;
            font-weight: normal;
        }}
        
        .cv-heatmap-{viewer_id} {{
            display: grid;
            gap: 0;
            background: #fff;
        }}
        
        .cv-heatmap-cell-{viewer_id} {{
            width: 4px;
            height: 4px;
        }}
        
        .cv-head-selector-{viewer_id} {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: flex-start;
        }}
        
        .cv-head-item-{viewer_id} {{
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
        }}
        
        .cv-head-thumb-{viewer_id} {{
            display: grid;
            gap: 0;
            background: #fff;
            border: 2px solid transparent;
            transition: border-color 0.15s;
        }}
        
        .cv-head-thumb-{viewer_id}:hover,
        .cv-head-thumb-{viewer_id}.active {{
            border-color: #666;
        }}
        
        .cv-head-thumb-cell-{viewer_id} {{
            width: 2px;
            height: 2px;
        }}
        
        .cv-head-label-{viewer_id} {{
            font-size: 11px;
            color: #ccc;
            margin-top: 4px;
        }}
        
        .cv-tokens-section-{viewer_id} {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #444;
        }}
        
        .cv-tokens-header-{viewer_id} {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        
        .cv-tokens-label-{viewer_id} {{
            font-size: 13px;
            font-weight: 600;
            color: #4a9eff;
        }}
        
        .cv-dropdown-{viewer_id} {{
            font-size: 12px;
            padding: 4px 8px;
            background: #444;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
        }}
        
        .cv-tokens-row-{viewer_id} {{
            display: flex;
            flex-wrap: wrap;
            gap: 0;
            background: #1a1a1a;
            padding: 8px;
            border-radius: 4px;
        }}
        
        .cv-token-{viewer_id} {{
            font-family: "SF Mono", "Monaco", "Inconsolata", "Fira Mono", monospace;
            font-size: 12px;
            padding: 2px 1px;
            cursor: pointer;
            transition: background-color 0.1s;
            background: #fff;
            color: #000;
            border: 1px solid transparent;
        }}
        
        .cv-token-{viewer_id}.selected {{
            border: 1px solid #4a9eff;
        }}

        .cv-token-{viewer_id}.locked {{
            border: 2px solid #4a9eff;
            padding: 1px 0px;
        }}

        .cv-token-{viewer_id}.first-seq {{
            font-weight: 700;
        }}

        .cv-accuracy-section-{viewer_id} {{
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .cv-accuracy-label-{viewer_id} {{
            font-size: 13px;
            font-weight: 600;
            color: #fff;
        }}

        .cv-accuracy-value-{viewer_id} {{
            font-size: 13px;
            font-weight: 600;
            color: #4a9eff;
        }}

        .cv-accuracy-percent-{viewer_id} {{
            font-size: 13px;
            color: #4a9eff;
            font-weight: 600;
        }}

        .cv-model-selector-{viewer_id} {{
            display: none;
            align-items: center;
            gap: 6px;
            margin-right: 16px;
        }}

        .cv-model-selector-{viewer_id}.visible {{
            display: flex;
        }}

        .cv-predictions-section-{viewer_id} {{
            margin-top: 12px;
            padding: 12px;
            background: #1a1a1a;
            border-radius: 4px;
            height: 140px;
            overflow: hidden;
        }}

        .cv-predictions-title-{viewer_id} {{
            font-size: 13px;
            font-weight: 600;
            color: #4a9eff;
            margin-bottom: 8px;
        }}

        .cv-predictions-list-{viewer_id} {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .cv-prediction-item-{viewer_id} {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }}

        .cv-prediction-rank-{viewer_id} {{
            color: #666;
            width: 16px;
        }}

        .cv-prediction-token-{viewer_id} {{
            font-family: "SF Mono", "Monaco", "Inconsolata", "Fira Mono", monospace;
            background: #333;
            padding: 2px 6px;
            border-radius: 3px;
            color: #fff;
            min-width: 30px;
            text-align: center;
        }}

        .cv-prediction-token-{viewer_id}.correct {{
            background: #2e7d32;
        }}

        .cv-prediction-prob-{viewer_id} {{
            color: #888;
            font-size: 11px;
        }}

        .cv-predictions-placeholder-{viewer_id} {{
            color: #666;
            font-size: 12px;
            font-style: italic;
        }}
    </style>
    
    <div class="cv-container-{viewer_id}">
        <div class="cv-accuracy-section-{viewer_id}">
            <div id="cv-model-selector-container-{viewer_id}" class="cv-model-selector-{viewer_id}">
                <span class="cv-section-title-{viewer_id}">Model:</span>
                <select id="cv-model-selector-{viewer_id}" class="cv-dropdown-{viewer_id}"></select>
            </div>
            <span class="cv-accuracy-label-{viewer_id}">Induction Accuracy:</span>
            <span id="cv-accuracy-value-{viewer_id}" class="cv-accuracy-value-{viewer_id}">-</span><span class="cv-accuracy-percent-{viewer_id}">%</span>
            <span style="margin-left: 16px;" class="cv-accuracy-label-{viewer_id}">Sequence Length:</span>
            <span class="cv-accuracy-value-{viewer_id}">{seq_len}</span>
        </div>
        <div class="cv-main-{viewer_id}">
            <div class="cv-left-{viewer_id}">
                <div class="cv-section-title-{viewer_id}" style="display: flex; gap: 8px;">
                    <span>Attention Pattern:</span>
                    <span id="cv-current-head-{viewer_id}" class="cv-accuracy-value-{viewer_id}"></span>
                </div>
                <div id="cv-main-heatmap-{viewer_id}" class="cv-heatmap-{viewer_id}"></div>
            </div>

            <div class="cv-right-{viewer_id}">
                <div class="cv-section-title-{viewer_id}">
                    Head selector <span class="cv-section-subtitle-{viewer_id}">(hover to focus, click to lock)</span>
                </div>
                <div id="cv-head-selector-{viewer_id}" class="cv-head-selector-{viewer_id}"></div>
            </div>
        </div>
        
        <div class="cv-tokens-section-{viewer_id}">
            <div class="cv-tokens-header-{viewer_id}">
                <span class="cv-tokens-label-{viewer_id}">Tokens</span>
                <span class="cv-section-subtitle-{viewer_id}">(hover to focus, click to lock focus)</span>
                <select id="cv-direction-{viewer_id}" class="cv-dropdown-{viewer_id}">
                    <option value="src">Source ← Destination</option>
                    <option value="dest">Destination ← Source</option>
                </select>
            </div>
            <div id="cv-tokens-row-{viewer_id}" class="cv-tokens-row-{viewer_id}"></div>
            <div class="cv-predictions-section-{viewer_id}">
                <div class="cv-predictions-title-{viewer_id}">Next Token Predictions</div>
                <div id="cv-predictions-list-{viewer_id}" class="cv-predictions-list-{viewer_id}">
                    <span class="cv-predictions-placeholder-{viewer_id}">Hover over a token to see predictions</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    (function() {{
        const tokens = {tokens_json};
        const allAttentionData = {all_attention_json};
        const allHeadLists = {all_head_lists_json};
        const allPredictions = {all_predictions_json};
        const allAccuracies = {all_accuracies_json};
        const modelNames = {model_names_json};
        const showModelDropdown = {show_model_dropdown_json};
        const viewerId = "{viewer_id}";
        const maxTokens = Math.min(tokens.length, {max_tokens});
        const seqLen = {seq_len};

        let currentModel = modelNames[0];
        let attentionData = allAttentionData[currentModel];
        let headList = allHeadLists[currentModel];
        let predictions = allPredictions[currentModel];
        let currentHead = headList[0];
        let lockedHead = null;
        let selectedToken = null;
        let lockedToken = null;
        let direction = 'src';  // 'src' = Source ← Destination (show what dest attends to)
        
        // Color interpolation for attention (white to dark red)
        function getColor(weight) {{
            const r = 255;
            const g = Math.round(255 * (1 - weight));
            const b = Math.round(255 * (1 - weight));
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}
        
        // Create main heatmap
        function renderMainHeatmap() {{
            const container = document.getElementById('cv-main-heatmap-' + viewerId);
            const pattern = attentionData[currentHead];
            const size = Math.min(pattern.length, maxTokens);
            
            container.style.gridTemplateColumns = `repeat(${{size}}, 4px)`;
            container.innerHTML = '';
            
            for (let i = 0; i < size; i++) {{
                for (let j = 0; j < size; j++) {{
                    const cell = document.createElement('div');
                    cell.className = 'cv-heatmap-cell-' + viewerId;
                    const weight = (j <= i) ? pattern[i][j] : 0;  // Causal mask
                    cell.style.backgroundColor = getColor(weight);
                    container.appendChild(cell);
                }}
            }}
        }}
        
        // Create head selector thumbnails
        function renderHeadSelector() {{
            const container = document.getElementById('cv-head-selector-' + viewerId);
            container.innerHTML = '';
            
            headList.forEach((headName) => {{
                const item = document.createElement('div');
                item.className = 'cv-head-item-' + viewerId;
                
                const thumb = document.createElement('div');
                thumb.className = 'cv-head-thumb-' + viewerId;
                if (headName === currentHead) thumb.classList.add('active');
                
                const pattern = attentionData[headName];
                const thumbSize = Math.min(pattern.length, 40);
                thumb.style.gridTemplateColumns = `repeat(${{thumbSize}}, 2px)`;
                
                for (let i = 0; i < thumbSize; i++) {{
                    for (let j = 0; j < thumbSize; j++) {{
                        const cell = document.createElement('div');
                        cell.className = 'cv-head-thumb-cell-' + viewerId;
                        const weight = (j <= i) ? pattern[i][j] : 0;
                        cell.style.backgroundColor = getColor(weight);
                        thumb.appendChild(cell);
                    }}
                }}
                
                const label = document.createElement('div');
                label.className = 'cv-head-label-' + viewerId;
                label.textContent = headName;
                
                thumb.onmouseenter = () => {{
                    if (!lockedHead) {{
                        currentHead = headName;
                        renderMainHeatmap();
                        updateTokenHighlights();
                        updateHeadSelection();
                    }}
                }};
                
                thumb.onclick = () => {{
                    if (lockedHead === headName) {{
                        lockedHead = null;
                    }} else {{
                        lockedHead = headName;
                        currentHead = headName;
                    }}
                    renderMainHeatmap();
                    updateTokenHighlights();
                    updateHeadSelection();
                }};
                
                item.appendChild(thumb);
                item.appendChild(label);
                container.appendChild(item);
            }});
        }}
        
        function updateHeadSelection() {{
            document.querySelectorAll('.cv-head-thumb-' + viewerId).forEach((thumb, idx) => {{
                thumb.classList.toggle('active', headList[idx] === currentHead);
            }});
            document.getElementById('cv-current-head-' + viewerId).textContent = currentHead;
        }}
        
        // Render predictions for a given token position
        function renderPredictions(tokenIdx) {{
            const container = document.getElementById('cv-predictions-list-' + viewerId);

            if (tokenIdx === null || tokenIdx >= predictions.length) {{
                container.innerHTML = '<span class="cv-predictions-placeholder-' + viewerId + '">Hover over a token to see predictions</span>';
                return;
            }}

            const pred = predictions[tokenIdx];
            const actualNextToken = tokens[tokenIdx + 1];
            container.innerHTML = '';

            pred.tokens.forEach((tok, rank) => {{
                const item = document.createElement('div');
                item.className = 'cv-prediction-item-' + viewerId;

                const rankEl = document.createElement('span');
                rankEl.className = 'cv-prediction-rank-' + viewerId;
                rankEl.textContent = (rank + 1) + '.';

                const tokenEl = document.createElement('span');
                tokenEl.className = 'cv-prediction-token-' + viewerId;
                tokenEl.textContent = tok;
                if (tok === actualNextToken) {{
                    tokenEl.classList.add('correct');
                }}

                const probEl = document.createElement('span');
                probEl.className = 'cv-prediction-prob-' + viewerId;
                probEl.textContent = (pred.probs[rank] * 100).toFixed(1) + '%';

                item.appendChild(rankEl);
                item.appendChild(tokenEl);
                item.appendChild(probEl);
                container.appendChild(item);
            }});
        }}

        // Create token row
        function renderTokens() {{
            const container = document.getElementById('cv-tokens-row-' + viewerId);
            container.innerHTML = '';

            tokens.slice(0, maxTokens).forEach((tok, idx) => {{
                const span = document.createElement('span');
                span.className = 'cv-token-' + viewerId;
                if (idx < seqLen) {{
                    span.classList.add('first-seq');
                }}
                span.textContent = tok;
                span.dataset.idx = idx;

                span.onmouseenter = () => {{
                    if (lockedToken === null) {{
                        selectedToken = idx;
                        updateTokenHighlights();
                        renderPredictions(idx);
                    }}
                }};

                span.onmouseleave = () => {{
                    if (lockedToken === null) {{
                        selectedToken = null;
                        updateTokenHighlights();
                        renderPredictions(null);
                    }}
                }};

                span.onclick = () => {{
                    if (lockedToken === idx) {{
                        // Unlock current token
                        lockedToken = null;
                        selectedToken = idx;  // Keep it selected since mouse is still over it
                    }} else {{
                        // Lock to this token (or switch to it)
                        lockedToken = idx;
                        selectedToken = idx;
                    }}
                    updateTokenHighlights();
                    renderPredictions(selectedToken);
                }};

                container.appendChild(span);
            }});
        }}
        
        function updateTokenHighlights() {{
            const tokenEls = document.querySelectorAll('.cv-token-' + viewerId);
            const pattern = attentionData[currentHead];
            
            tokenEls.forEach((el, idx) => {{
                el.classList.remove('selected');
                el.classList.remove('locked');
                el.style.backgroundColor = '#fff';
                el.style.color = '#000';
                
                if (selectedToken !== null) {{
                    let weight = 0;
                    if (direction === 'src') {{
                        // Source ← Destination: selectedToken is destination, show source weights
                        if (idx <= selectedToken && pattern[selectedToken]) {{
                            weight = pattern[selectedToken][idx];
                        }}
                    }} else {{
                        // Destination ← Source: selectedToken is source, show who attends to it
                        if (idx >= selectedToken && pattern[idx]) {{
                            weight = pattern[idx][selectedToken];
                        }}
                    }}

                    if (weight > 0.01) {{
                        el.style.backgroundColor = getColor(weight);
                        el.style.color = weight > 0.5 ? '#fff' : '#000';
                    }}

                    if (idx === selectedToken) {{
                        el.classList.add(lockedToken === idx ? 'locked' : 'selected');
                    }}
                }}
            }});
        }}
        
        // Direction dropdown
        document.getElementById('cv-direction-' + viewerId).onchange = (e) => {{
            direction = e.target.value;
            updateTokenHighlights();
        }};

        // Update accuracy display
        function updateAccuracyDisplay() {{
            const accuracy = allAccuracies[currentModel];
            document.getElementById('cv-accuracy-value-' + viewerId).textContent = accuracy;
        }}

        // Model selector setup
        function setupModelSelector() {{
            const container = document.getElementById('cv-model-selector-container-' + viewerId);
            const selector = document.getElementById('cv-model-selector-' + viewerId);

            if (showModelDropdown) {{
                container.classList.add('visible');

                // Populate dropdown
                modelNames.forEach((name) => {{
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name;
                    selector.appendChild(option);
                }});

                // Change handler
                selector.onchange = (e) => {{
                    currentModel = e.target.value;
                    attentionData = allAttentionData[currentModel];
                    headList = allHeadLists[currentModel];
                    predictions = allPredictions[currentModel];
                    currentHead = headList[0];
                    lockedHead = null;
                    renderHeadSelector();
                    renderMainHeatmap();
                    updateTokenHighlights();
                    updateAccuracyDisplay();
                    renderPredictions(selectedToken);
                }};
            }}
        }}

        // Initial render
        setupModelSelector();
        updateAccuracyDisplay();
        updateHeadSelection();
        renderHeadSelector();
        renderMainHeatmap();
        renderTokens();
    }})();
    </script>
    """
    
    display(HTML(html))
    return None



def interactive_training_viewer(history: dict, max_snapshots: int = 200, attn_resolution: int = 50):
    """
    Create an interactive training visualization with:
    - Accuracy and loss plots on the left
    - Attention head heatmaps on the right
    - Epoch slider to explore training dynamics
    
    Args:
        history: Training history dict
        max_snapshots: Maximum number of attention snapshots to include (subsampled evenly)
        attn_resolution: Downsample attention patterns to this resolution
    """
    # Extract data
    epochs = history['epochs']
    losses = history['losses']
    induction_accs = history['induction_accs']
    ood_accs = history['ood_induction_accs']
    attn_snapshots = history['attention_snapshots']
    
    # Get sorted attention epochs and subsample
    all_attn_epochs = sorted(attn_snapshots.keys())
    if len(all_attn_epochs) > max_snapshots:
        step = len(all_attn_epochs) // max_snapshots
        attn_epochs = all_attn_epochs[::step][:max_snapshots]
    else:
        attn_epochs = all_attn_epochs
    
    print(f"Using {len(attn_epochs)} attention snapshots (subsampled from {len(all_attn_epochs)})")
    
    # Prepare attention data - convert tensors to lists with downsampling
    attn_data = {}
    for epoch in attn_epochs:
        attn = attn_snapshots[epoch]  # [n_layers, n_heads, seq, seq]
        if hasattr(attn, 'cpu'):
            attn = attn.cpu().numpy()
        
        # Downsample attention patterns if needed
        seq_len = attn.shape[-1]
        if seq_len > attn_resolution:
            # Simple downsampling by taking every nth element
            step = seq_len // attn_resolution
            attn = attn[:, :, ::step, ::step][:, :, :attn_resolution, :attn_resolution]
        
        attn_data[int(epoch)] = attn.tolist()
    
    # Prepare plot data
    plot_data = {
        'epochs': epochs,
        'losses': losses,
        'in_dist_acc': induction_accs,
        'ood_accs': {str(k): v for k, v in ood_accs.items()},
        'attn_epochs': [int(e) for e in attn_epochs],
    }
    
    viewer_id = "training_viewer"
    plot_data_json = json.dumps(plot_data)
    attn_data_json = json.dumps(attn_data)
    
    html = f"""
    <style>
        .tv-container-{viewer_id} {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            color: #333;
            width: 100%;
            box-sizing: border-box;
        }}
        
        .tv-container-{viewer_id} * {{
            font-family: inherit;
        }}
        
        .tv-header-{viewer_id} {{
            text-align: center;
            margin-bottom: 16px;
        }}
        
        .tv-title-{viewer_id} {{
            font-size: 20px;
            font-weight: 600;
            color: #1a5fb4;
        }}
        
        .tv-controls-{viewer_id} {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 20px;
            padding: 12px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .tv-slider-{viewer_id} {{
            width: 400px;
            cursor: pointer;
        }}
        
        .tv-epoch-display-{viewer_id} {{
            font-size: 14px;
            min-width: 120px;
        }}
        
        .tv-btn-{viewer_id} {{
            padding: 6px 12px;
            border: 1px solid #ccc;
            background: #fff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}
        
        .tv-btn-{viewer_id}:hover {{
            background: #f0f0f0;
        }}
        
        .tv-main-{viewer_id} {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .tv-row-{viewer_id} {{
            display: flex;
            gap: 16px;
            align-items: flex-start;
            justify-content: center;
        }}
        
        .tv-plot-container-{viewer_id} {{
            background: #f5f5f5;
            border-radius: 4px;
            padding: 8px;
            position: relative;
        }}
        
        .tv-plot-title-{viewer_id} {{
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }}
        
        .tv-canvas-{viewer_id} {{
            display: block;
        }}
        
        .tv-legend-{viewer_id} {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 6px;
            font-size: 11px;
        }}
        
        .tv-legend-item-{viewer_id} {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        
        .tv-legend-color-{viewer_id} {{
            width: 20px;
            height: 3px;
            border-radius: 1px;
        }}
        
        .tv-layer-section-{viewer_id} {{
            background: #fff;
            padding: 6px 8px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .tv-layer-title-{viewer_id} {{
            font-size: 12px;
            font-weight: 600;
            color: #333;
            margin-bottom: 6px;
        }}
        
        .tv-heads-grid-{viewer_id} {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2px;
        }}
        
        .tv-head-container-{viewer_id} {{
            text-align: center;
        }}
        
        .tv-head-label-{viewer_id} {{
            font-size: 10px;
            color: #888;
            margin-top: 4px;
        }}
        
        .tv-attn-canvas-{viewer_id} {{
            image-rendering: pixelated;
        }}
    </style>
    
    <div class="tv-container-{viewer_id}">
        <div class="tv-header-{viewer_id}">
            <div class="tv-title-{viewer_id}">Training A 2-Layer Transformer On Repeating Token Sequences</div>
        </div>
        
        <div class="tv-controls-{viewer_id}">
            <button class="tv-btn-{viewer_id}" id="tv-play-{viewer_id}">▶ Play</button>
            <input type="range" class="tv-slider-{viewer_id}" id="tv-slider-{viewer_id}" min="0" max="100" value="0">
            <span class="tv-epoch-display-{viewer_id}" id="tv-epoch-{viewer_id}">Epoch: 0</span>
        </div>
        
        <div class="tv-main-{viewer_id}">
            <div class="tv-row-{viewer_id}">
                <div class="tv-plot-container-{viewer_id}">
                    <canvas id="tv-acc-canvas-{viewer_id}" class="tv-canvas-{viewer_id}" width="700" height="280"></canvas>
                    <div class="tv-legend-{viewer_id}" id="tv-acc-legend-{viewer_id}"></div>
                </div>
                <div class="tv-layer-section-{viewer_id}" id="tv-layer0-{viewer_id}"></div>
            </div>
            <div class="tv-row-{viewer_id}">
                <div class="tv-plot-container-{viewer_id}">
                    <canvas id="tv-loss-canvas-{viewer_id}" class="tv-canvas-{viewer_id}" width="700" height="280"></canvas>
                </div>
                <div class="tv-layer-section-{viewer_id}" id="tv-layer1-{viewer_id}"></div>
            </div>
        </div>
    </div>
    
    <script>
    (function() {{
        const viewerId = "{viewer_id}";
        const plotData = {plot_data_json};
        const attnData = {attn_data_json};
        
        const epochs = plotData.epochs;
        const attnEpochs = plotData.attn_epochs;
        let currentIdx = 0;
        let playing = false;
        let playInterval = null;
        
        const colors = {{
            'in_dist': '#2ca02c',
            '10': '#9467bd',
            '15': '#ff7f0e', 
            '35': '#17becf',
            '40': '#d62728'
        }};
        
        const slider = document.getElementById('tv-slider-' + viewerId);
        slider.max = attnEpochs.length - 1;
        
        // Draw accuracy plot
        function drawAccuracyPlot(currentEpoch) {{
            const canvas = document.getElementById('tv-acc-canvas-' + viewerId);
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const margin = {{left: 50, right: 20, top: 20, bottom: 40}};
            const plotW = W - margin.left - margin.right;
            const plotH = H - margin.top - margin.bottom;
            
            ctx.clearRect(0, 0, W, H);
            
            // Background
            ctx.fillStyle = '#f5f5f5';
            ctx.fillRect(margin.left, margin.top, plotW, plotH);
            
            // Grid
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {{
                const y = margin.top + (plotH * i / 5);
                ctx.beginPath();
                ctx.moveTo(margin.left, y);
                ctx.lineTo(W - margin.right, y);
                ctx.stroke();
            }}
            
            const maxEpoch = epochs[epochs.length - 1];
            const xScale = (e) => margin.left + (e / maxEpoch) * plotW;
            const yScale = (v) => margin.top + plotH - (v * plotH);
            
            // Draw lines
            function drawLine(data, color) {{
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (let i = 0; i < data.length; i++) {{
                    const x = xScale(epochs[i]);
                    const y = yScale(data[i]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();
            }}
            
            drawLine(plotData.in_dist_acc, colors.in_dist);
            Object.keys(plotData.ood_accs).forEach(k => {{
                drawLine(plotData.ood_accs[k], colors[k]);
            }});
            
            // Current epoch line
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            const cx = xScale(currentEpoch);
            ctx.beginPath();
            ctx.moveTo(cx, margin.top);
            ctx.lineTo(cx, margin.top + plotH);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Axes
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, margin.top + plotH);
            ctx.lineTo(W - margin.right, margin.top + plotH);
            ctx.stroke();
            
            // Labels
            ctx.fillStyle = '#666';
            ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Epoch', W/2, H - 5);
            
            ctx.save();
            ctx.translate(15, H/2);
            ctx.rotate(-Math.PI/2);
            ctx.fillText('Accuracy', 0, 0);
            ctx.restore();
            
            // Y axis labels
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {{
                const v = i / 5;
                const y = margin.top + plotH - (v * plotH);
                ctx.fillText(v.toFixed(1), margin.left - 5, y + 4);
            }}
            
            // X axis labels
            ctx.textAlign = 'center';
            for (let i = 0; i <= 5; i++) {{
                const e = Math.round(maxEpoch * i / 5);
                const x = xScale(e);
                ctx.fillText((e/1000).toFixed(0) + 'k', x, margin.top + plotH + 15);
            }}
        }}
        
        // Draw loss plot
        function drawLossPlot(currentEpoch) {{
            const canvas = document.getElementById('tv-loss-canvas-' + viewerId);
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const margin = {{left: 50, right: 20, top: 20, bottom: 40}};
            const plotW = W - margin.left - margin.right;
            const plotH = H - margin.top - margin.bottom;
            
            ctx.clearRect(0, 0, W, H);
            
            // Background
            ctx.fillStyle = '#f5f5f5';
            ctx.fillRect(margin.left, margin.top, plotW, plotH);
            
            const maxEpoch = epochs[epochs.length - 1];
            const maxLoss = Math.max(...plotData.losses);
            const xScale = (e) => margin.left + (e / maxEpoch) * plotW;
            const yScale = (v) => margin.top + plotH - (v / maxLoss) * plotH;
            
            // Grid
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {{
                const y = margin.top + (plotH * i / 5);
                ctx.beginPath();
                ctx.moveTo(margin.left, y);
                ctx.lineTo(W - margin.right, y);
                ctx.stroke();
            }}
            
            // Loss line
            ctx.strokeStyle = '#d62728';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < plotData.losses.length; i++) {{
                const x = xScale(epochs[i]);
                const y = yScale(plotData.losses[i]);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();
            
            // Current epoch line
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            const cx = xScale(currentEpoch);
            ctx.beginPath();
            ctx.moveTo(cx, margin.top);
            ctx.lineTo(cx, margin.top + plotH);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Axes
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, margin.top + plotH);
            ctx.lineTo(W - margin.right, margin.top + plotH);
            ctx.stroke();
            
            // Labels
            ctx.fillStyle = '#666';
            ctx.font = '11px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Epoch', W/2, H - 5);
            
            ctx.save();
            ctx.translate(15, H/2);
            ctx.rotate(-Math.PI/2);
            ctx.fillText('Loss', 0, 0);
            ctx.restore();
            
            // Y axis labels
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {{
                const v = maxLoss * (5 - i) / 5;
                const y = margin.top + (plotH * i / 5);
                ctx.fillText(v.toFixed(1), margin.left - 5, y + 4);
            }}
            
            // X axis labels
            ctx.textAlign = 'center';
            for (let i = 0; i <= 5; i++) {{
                const e = Math.round(maxEpoch * i / 5);
                const x = xScale(e);
                ctx.fillText((e/1000).toFixed(0) + 'k', x, margin.top + plotH + 15);
            }}
        }}
        
        // Draw attention heatmaps for a single layer
        function drawLayerAttention(layer, epoch, containerId) {{
            const container = document.getElementById(containerId);
            const attn = attnData[epoch];
            
            if (!attn || !attn[layer]) return;
            
            const nHeads = attn[layer].length;
            const seqLen = attn[layer][0].length;
            const canvasSize = 135;  // Display size for each attention head
            
            container.innerHTML = '';
            
            const title = document.createElement('div');
            title.className = 'tv-layer-title-' + viewerId;
            title.textContent = `Layer ${{layer}} attention (seq_len=20)`;
            container.appendChild(title);
            
            const grid = document.createElement('div');
            grid.className = 'tv-heads-grid-' + viewerId;
            
            for (let head = 0; head < nHeads; head++) {{
                const headContainer = document.createElement('div');
                headContainer.className = 'tv-head-container-' + viewerId;
                
                const canvas = document.createElement('canvas');
                canvas.className = 'tv-attn-canvas-' + viewerId;
                canvas.width = seqLen;
                canvas.height = seqLen;
                canvas.style.width = canvasSize + 'px';
                canvas.style.height = canvasSize + 'px';
                
                const ctx = canvas.getContext('2d');
                const pattern = attn[layer][head];
                
                for (let i = 0; i < seqLen; i++) {{
                    for (let j = 0; j < seqLen; j++) {{
                        const weight = (j <= i) ? pattern[i][j] : 0;
                        const intensity = Math.round(255 * (1 - weight));
                        ctx.fillStyle = `rgb(255, ${{intensity}}, ${{intensity}})`;
                        ctx.fillRect(j, i, 1, 1);
                    }}
                }}
                
                headContainer.appendChild(canvas);
                grid.appendChild(headContainer);
            }}
            
            container.appendChild(grid);
        }}
        
        // Draw attention heatmaps for both layers
        function drawAttentionHeatmaps(epoch) {{
            drawLayerAttention(0, epoch, 'tv-layer0-' + viewerId);
            drawLayerAttention(1, epoch, 'tv-layer1-' + viewerId);
        }}
        
        // Update display
        function update(idx) {{
            currentIdx = idx;
            const epoch = attnEpochs[idx];
            
            document.getElementById('tv-epoch-' + viewerId).textContent = `Epoch: ${{epoch}}`;
            slider.value = idx;
            
            drawAccuracyPlot(epoch);
            drawLossPlot(epoch);
            drawAttentionHeatmaps(epoch);
        }}
        
        // Create legend
        function createLegend() {{
            const legend = document.getElementById('tv-acc-legend-' + viewerId);
            const items = [
                ['In-distribution (seq_len=20-30)', colors.in_dist],
                ['OOD seq_len=10', colors['10']],
                ['OOD seq_len=15', colors['15']],
                ['OOD seq_len=35', colors['35']],
                ['OOD seq_len=40', colors['40']]
            ];
            
            items.forEach(([label, color]) => {{
                const item = document.createElement('div');
                item.className = 'tv-legend-item-' + viewerId;
                item.innerHTML = `<div class="tv-legend-color-${{viewerId}}" style="background:${{color}}"></div>${{label}}`;
                legend.appendChild(item);
            }});
        }}
        
        // Controls
        slider.oninput = (e) => update(parseInt(e.target.value));
        
        const playBtn = document.getElementById('tv-play-' + viewerId);
        playBtn.onclick = () => {{
            if (playing) {{
                playing = false;
                playBtn.textContent = '▶ Play';
                clearInterval(playInterval);
            }} else {{
                playing = true;
                playBtn.textContent = '⏸ Pause';
                playInterval = setInterval(() => {{
                    currentIdx = (currentIdx + 1) % attnEpochs.length;
                    update(currentIdx);
                }}, 100);
            }}
        }};
        
        // Initial render
        createLegend();
        update(0);
    }})();
    </script>
    """
    
    display(HTML(html))
