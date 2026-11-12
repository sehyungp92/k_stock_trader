"""Check exact holding continuation before the native audit adds its identity.

No trading attribute or audit field is excluded. The complete post-audit state
must equal the pre-audit state plus the exact published evaluation identity.
"""
from copy import deepcopy
import hashlib

from .holding_account_replay_v1 import HoldingAccountReplay as PreviousReplay
from . import account_fork_v2 as states
from .pending_entry_account_pilot_v1 import canonical, neutral
from .retention_research_io_v1 import write


class HoldingAccountReplay(PreviousReplay):
    def install(self):
        super().install()
        from backtests.engine import integrity
        self.old_audit=integrity.audit_broker
        self.audit_transition=None
        def audit(broker, *, identity=None):
            if self.parent is None or broker is not self.parent[1]:
                return self.old_audit(broker,identity=identity)
            # Both states now refer to the same fully processed EOD boundary.
            # The existing full-state equality requirement is unchanged.
            self.finish_day()
            before=states.receipt(*self.parent)
            result=self.old_audit(broker,identity=identity)
            after=states.receipt(*self.parent)
            expected=deepcopy(before['state'])
            expected['broker']['evaluation_identity']=neutral(result['audit_receipt']['identity'])
            expected_hash=hashlib.sha256(canonical(expected).encode()).hexdigest()
            if after['state']!=expected or after['sha256']!=expected_hash:
                raise ValueError('Native audit changed state beyond its exact published identity')
            self.audit_transition={'passed':True,'before_sha256':before['sha256'],
                'after_sha256':after['sha256'],'expected_after_sha256':expected_hash,
                'published_evaluation_identity':expected['broker']['evaluation_identity'],
                'scope':'Complete state checked; only the exact native audit identity is added.'}
            write(self.output/'terminal_audit_transition.json',self.audit_transition)
            return result
        integrity.audit_broker=audit

    def restore(self):
        from backtests.engine import integrity
        integrity.audit_broker=self.old_audit
        super().restore()

    def finish(self):
        if self.audit_transition is None:
            raise ValueError('Missing verified native terminal audit transition')
        if states.receipt(*self.parent)['sha256']!=self.audit_transition['after_sha256']:
            raise ValueError('State changed after the verified terminal audit')
        super().finish()
