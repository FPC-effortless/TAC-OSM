        query: Query,
        pool: Sequence[AddressedMemory],
        *,
        target_address: str | None = None,
        total_macs: int | None = None,
    ) -> StateAddressingDecision:
        if not pool:
            raise ValueError("semantic state pool is empty")
        if total_macs is None:
            scores, macs = self.score_pool(query, pool)
        else:
            scores = self.score_pool(query, pool)[0]
            macs = total_macs
        selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
        rank = 1
        if target_address is not None:
            target_indices = [
                i for i, item in enumerate(pool) if item.address == target_address
            ]
            if len(target_indices) != 1:
                raise ValueError("target address absent or duplicated in state pool")
            target_score = scores[target_indices[0]]
            rank = 1 + sum(score > target_score for score in scores)
        return StateAddressingDecision(
            selected_index=selected,
            selected_address=pool[selected].address,
            scores=tuple(scores),
            target_rank=rank,
            inspected_items=len(pool),
            pool_size=len(pool),
            total_macs=macs,
        )

    def select(
        self,
        query: Query,
        state: TemporalPersistentState,
        *,
        target_address: str | None = None,
    ) -> StateAddressingDecision:
        pool = self.state_pool(query, state)
        return self.select_from_pool(
            query,
            pool,
            target_address=target_address,
        )

    def set_identity(self) -> None:
        for r in range(self.config.latent_dim):
            for j in range(self.config.input_dim):
                value = 1.0 if r == j else 0.0
                self.wq[r][j] = value
                self.ws[r][j] = value
            self.bq[r] = 0.0