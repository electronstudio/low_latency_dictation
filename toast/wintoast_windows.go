//go:build windows

// WinRT COM vtable definitions adapted from git.sr.ht/~jackmordaunt/go-toast/v2
// (MIT / Unlicense). The originals are auto-generated from Windows SDK metadata.

package toast

import (
	"fmt"
	"html"
	"syscall"
	"unsafe"

	"github.com/go-ole/go-ole"
	"golang.org/x/sys/windows/registry"
)

// ---- ToastNotificationManager statics ----

const guidToastNotificationManagerStatics5 = "{d6f5f569-d40d-407c-8989-88cab42cfd14}"

type iToastNotificationManagerStatics5 struct{ ole.IInspectable }
type iToastNotificationManagerStatics5Vtbl struct {
	ole.IInspectableVtbl
	GetDefault uintptr
}

func (v *iToastNotificationManagerStatics5) VTable() *iToastNotificationManagerStatics5Vtbl {
	return (*iToastNotificationManagerStatics5Vtbl)(unsafe.Pointer(v.RawVTable))
}

type toastNotificationManagerForUser struct{ ole.IUnknown }

// getDefaultManager returns the ToastNotificationManagerForUser.
func getDefaultManager() (*toastNotificationManagerForUser, error) {
	inspectable, err := ole.RoGetActivationFactory(
		"Windows.UI.Notifications.ToastNotificationManager",
		ole.NewGUID(guidToastNotificationManagerStatics5))
	if err != nil {
		return nil, err
	}
	defer inspectable.Release() // factory is a static; release our ref
	v := (*iToastNotificationManagerStatics5)(unsafe.Pointer(inspectable))
	var out *toastNotificationManagerForUser
	hr, _, _ := syscall.SyscallN(v.VTable().GetDefault, 0, uintptr(unsafe.Pointer(&out)))
	if hr != 0 {
		return nil, ole.NewError(hr)
	}
	return out, nil
}

// ---- ToastNotificationManagerForUser ----

const guidToastNotificationManagerForUser = "{79ab57f6-43fe-487b-8a7f-99567200ae94}"

type iToastNotificationManagerForUser struct{ ole.IInspectable }
type iToastNotificationManagerForUserVtbl struct {
	ole.IInspectableVtbl
	CreateToastNotifier       uintptr
	CreateToastNotifierWithId uintptr
	GetHistory                uintptr
	GetUser                   uintptr
}

func (v *iToastNotificationManagerForUser) VTable() *iToastNotificationManagerForUserVtbl {
	return (*iToastNotificationManagerForUserVtbl)(unsafe.Pointer(v.RawVTable))
}

func (m *toastNotificationManagerForUser) createToastNotifierWithId(appID string) (*toastNotifier, error) {
	itf := m.MustQueryInterface(ole.NewGUID(guidToastNotificationManagerForUser))
	defer itf.Release()
	v := (*iToastNotificationManagerForUser)(unsafe.Pointer(itf))
	hApp, err := ole.NewHString(appID)
	if err != nil {
		return nil, err
	}
	defer ole.DeleteHString(hApp)
	var out *toastNotifier
	hr, _, _ := syscall.SyscallN(
		v.VTable().CreateToastNotifierWithId,
		uintptr(unsafe.Pointer(v)),
		uintptr(hApp),
		uintptr(unsafe.Pointer(&out)),
	)
	if hr != 0 {
		return nil, ole.NewError(hr)
	}
	return out, nil
}

// ---- ToastNotifier ----

const guidToastNotifier = "{75927b93-03f3-41ec-91d3-6e5bac1b38e7}"

type toastNotifier struct{ ole.IUnknown }
type iToastNotifier struct{ ole.IInspectable }
type iToastNotifierVtbl struct {
	ole.IInspectableVtbl
	Show                           uintptr
	Hide                           uintptr
	GetSetting                     uintptr
	AddToSchedule                  uintptr
	RemoveFromSchedule             uintptr
	GetScheduledToastNotifications uintptr
}

func (v *iToastNotifier) VTable() *iToastNotifierVtbl {
	return (*iToastNotifierVtbl)(unsafe.Pointer(v.RawVTable))
}

func (n *toastNotifier) Show(t *toastNotification) error {
	itf := n.MustQueryInterface(ole.NewGUID(guidToastNotifier))
	defer itf.Release()
	v := (*iToastNotifier)(unsafe.Pointer(itf))
	hr, _, _ := syscall.SyscallN(v.VTable().Show,
		uintptr(unsafe.Pointer(v)), uintptr(unsafe.Pointer(t)))
	if hr != 0 {
		return ole.NewError(hr)
	}
	return nil
}

func (n *toastNotifier) Hide(t *toastNotification) error {
	itf := n.MustQueryInterface(ole.NewGUID(guidToastNotifier))
	defer itf.Release()
	v := (*iToastNotifier)(unsafe.Pointer(itf))
	hr, _, _ := syscall.SyscallN(v.VTable().Hide,
		uintptr(unsafe.Pointer(v)), uintptr(unsafe.Pointer(t)))
	if hr != 0 {
		return ole.NewError(hr)
	}
	return nil
}

func (n *toastNotifier) Release() { n.IUnknown.Release() }

// ---- ToastNotification ----

const guidToastNotification = "{997e2675-059e-4e60-8b06-1760917c8b80}"
const guidToastNotificationFactory = "{04124b20-82c6-4229-b109-fd9ed4662b53}"

type toastNotification struct{ ole.IUnknown }

func (t *toastNotification) Release() { t.IUnknown.Release() }

type iToastNotification struct{ ole.IInspectable }
type iToastNotificationVtbl struct {
	ole.IInspectableVtbl
	GetContent        uintptr
	SetExpirationTime uintptr
	GetExpirationTime uintptr
	AddDismissed      uintptr
	RemoveDismissed   uintptr
	AddActivated      uintptr
	RemoveActivated   uintptr
	AddFailed         uintptr
	RemoveFailed      uintptr
}

type iToastNotificationFactory struct{ ole.IInspectable }
type iToastNotificationFactoryVtbl struct {
	ole.IInspectableVtbl
	CreateToastNotification uintptr
}

func (v *iToastNotificationFactory) VTable() *iToastNotificationFactoryVtbl {
	return (*iToastNotificationFactoryVtbl)(unsafe.Pointer(v.RawVTable))
}

func createToastNotification(doc *xmlDocument) (*toastNotification, error) {
	inspectable, err := ole.RoGetActivationFactory(
		"Windows.UI.Notifications.ToastNotification",
		ole.NewGUID(guidToastNotificationFactory))
	if err != nil {
		return nil, err
	}
	defer inspectable.Release()
	v := (*iToastNotificationFactory)(unsafe.Pointer(inspectable))
	var out *toastNotification
	hr, _, _ := syscall.SyscallN(
		v.VTable().CreateToastNotification,
		0, // static method, no `this`
		uintptr(unsafe.Pointer(doc)),
		uintptr(unsafe.Pointer(&out)),
	)
	if hr != 0 {
		return nil, ole.NewError(hr)
	}
	return out, nil
}

// ---- XmlDocument ----

const guidXmlDocumentIO = "{6cd0e74e-ee65-4489-9ebf-ca43e87ba637}"

type xmlDocument struct{ ole.IUnknown }
type iXmlDocumentIO struct{ ole.IInspectable }
type iXmlDocumentIOVtbl struct {
	ole.IInspectableVtbl
	LoadXml             uintptr
	LoadXmlWithSettings uintptr
	SaveToFileAsync     uintptr
}

func (v *iXmlDocumentIO) VTable() *iXmlDocumentIOVtbl {
	return (*iXmlDocumentIOVtbl)(unsafe.Pointer(v.RawVTable))
}

func newXmlDocument() (*xmlDocument, error) {
	inspectable, err := ole.RoActivateInstance("Windows.Data.Xml.Dom.XmlDocument")
	if err != nil {
		return nil, err
	}
	return (*xmlDocument)(unsafe.Pointer(inspectable)), nil
}

func (d *xmlDocument) LoadXml(xml string) error {
	itf := d.MustQueryInterface(ole.NewGUID(guidXmlDocumentIO))
	defer itf.Release()
	v := (*iXmlDocumentIO)(unsafe.Pointer(itf))
	hXml, err := ole.NewHString(xml)
	if err != nil {
		return err
	}
	defer ole.DeleteHString(hXml)
	hr, _, _ := syscall.SyscallN(v.VTable().LoadXml,
		uintptr(unsafe.Pointer(v)), uintptr(hXml))
	if hr != 0 {
		return ole.NewError(hr)
	}
	return nil
}

func (d *xmlDocument) Release() { d.IUnknown.Release() }

// ---- Orchestrator helpers ----

// newNotifier creates and returns the ToastNotifier for the given AUMID.
func newNotifier(appID string) (*toastNotifier, error) {
	mgr, err := getDefaultManager()
	if err != nil {
		return nil, fmt.Errorf("getDefaultManager: %w", err)
	}
	defer mgr.Release()
	return mgr.createToastNotifierWithId(appID)
}

// createAndShow builds the doc, creates the toast, shows it, and returns the
// retained toast handle (caller must Release it later).
func createAndShow(n *toastNotifier, xml string) (*toastNotification, error) {
	doc, err := newXmlDocument()
	if err != nil {
		return nil, fmt.Errorf("newXmlDocument: %w", err)
	}
	defer doc.Release()
	if err := doc.LoadXml(xml); err != nil {
		return nil, fmt.Errorf("LoadXml: %w", err)
	}
	t, err := createToastNotification(doc)
	if err != nil {
		return nil, fmt.Errorf("createToastNotification: %w", err)
	}
	if err := n.Show(t); err != nil {
		t.Release()
		return nil, fmt.Errorf("Show: %w", err)
	}
	return t, nil
}

// buildToastXML constructs a silent ToastGeneric payload.
func buildToastXML(title, body, duration string) string {
	t := html.EscapeString(title)
	if body == "" {
		return fmt.Sprintf(`<toast duration="%s"><visual><binding template="ToastGeneric"><text>%s</text></binding></visual><audio silent="true"/></toast>`, duration, t)
	}
	b := html.EscapeString(body)
	return fmt.Sprintf(`<toast duration="%s"><visual><binding template="ToastGeneric"><text>%s</text><text>%s</text></binding></visual><audio silent="true"/></toast>`, duration, t, b)
}

// registerAppData writes the AUMID to the registry so Action Center shows the
// app name. Display-only toasts need only DisplayName (and optionally IconUri).
func registerAppData(appID string) error {
	key := `SOFTWARE\Classes\AppUserModelId\` + appID
	k, _, err := registry.CreateKey(registry.CURRENT_USER, key, registry.SET_VALUE|registry.CREATE_SUB_KEY)
	if err != nil {
		return fmt.Errorf("open registry key: %w", err)
	}
	defer k.Close()
	if err := k.SetStringValue("DisplayName", appID); err != nil {
		return fmt.Errorf("set DisplayName: %w", err)
	}
	return nil
}
