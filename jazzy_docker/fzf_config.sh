#!/bin/bash
# fzf_config.sh

cat >> /root/.bashrc << 'EOF'

# FZF History Search
fzf_history() {
    local selected
    selected=$(history | tac | sed 's/^[ ]*[0-9]*[ ]*//' | fzf --height=40% --reverse --query="$READLINE_LINE")
    READLINE_LINE="$selected"
    READLINE_POINT=${#selected}
}

# Bind Ctrl+R to fzf history search
bind -x '"\C-r": fzf_history'

EOF